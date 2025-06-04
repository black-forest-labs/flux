import argparse
import torch
import os
import json
import math
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import gc

import sys
# 添加flux模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flux.util import load_flow_model, load_t5, load_clip, load_ae, configs
from flux.sampling import get_noise, prepare, get_schedule, denoise, unpack
from flux.model import Flux


def prepare_coco_text_and_image(json_file: str, max_samples: int = None) -> tuple[List[str], List[str]]:
    """
    准备COCO数据集的文本和图像路径
    
    Args:
        json_file: COCO标注文件路径
        max_samples: 最大样本数量
    
    Returns:
        (captions, image_paths): 文本描述列表和图像路径列表
    """
    with open(json_file, 'r') as f:
        info = json.load(f)
    
    annotation_list = info["annotations"]
    image_caption_dict = {}
    
    for annotation_dict in annotation_list:
        image_id = annotation_dict["image_id"]
        if image_id in image_caption_dict:
            image_caption_dict[image_id].append(annotation_dict["caption"])
        else:
            image_caption_dict[image_id] = [annotation_dict["caption"]]
    
    captions = []
    image_paths = []
    
    # 获取第一个caption和对应的图像路径
    for image_id, texts in image_caption_dict.items():
        captions.append(texts[0])
        # 假设图像存储在val2014目录下
        image_paths.append(f"COCO_val2014_{image_id:012}.jpg")
        
        if max_samples and len(captions) >= max_samples:
            break
    
    return captions, image_paths


def save_intermediate_states_with_offload(
    model: Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: List[float],
    guidance: float = 0.0,
    offload: bool = True,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    保存Flux模型推理过程中的中间状态，使用offload策略管理内存
    
    Args:
        model: Flux模型
        img: 图像潜在表示
        img_ids: 图像位置编码
        txt: 文本嵌入
        txt_ids: 文本位置编码
        vec: CLIP向量嵌入
        timesteps: 时间步列表
        guidance: 引导强度
        offload: 是否使用offload策略
        device: 设备
    
    Returns:
        包含中间状态的字典
    """
    trajectory = {}
    outputs = {}
    torch_device = torch.device(device)
    
    # 确保模型在评估模式
    model.eval()
    
    with torch.no_grad():
        # 保存初始噪声
        trajectory[timesteps[0]] = img.clone().cpu()  # 保存到CPU以节省GPU内存
        
        # 如果使用offload，将模型移到GPU
        if offload:
            model = model.to(torch_device)
            torch.cuda.empty_cache()
        
        # 逐步去噪
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            
            # 预测噪声
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
                y=vec,
                guidance=torch.full_like(t_vec, guidance) if model.params.guidance_embed else None,
            )
            
            # 更新图像（简化的欧拉积分）
            dt = t_prev - t_curr
            img = img + pred * dt
            
            # 保存中间状态到CPU以节省GPU内存
            trajectory[t_prev] = img.clone().cpu()
            outputs[t_prev] = pred.clone().cpu()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 如果使用offload，将模型移回CPU
        if offload:
            model = model.cpu()
            torch.cuda.empty_cache()
    
    return {
        'trajectory': trajectory,
        'outputs': outputs
    }


def generate_flux_calibration_data_with_offload(
    prompts: List[str],
    model_name: str = "flux-dev",
    num_steps: int = 4,
    guidance: float = 0.0,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    device: str = "cuda",
    offload: bool = True
) -> Dict[str, Any]:
    """
    为Flux模型生成校准数据，使用offload策略管理大模型内存
    
    Args:
        prompts: 文本提示列表
        model_name: 模型名称
        num_steps: 推理步数
        guidance: 引导强度
        height: 图像高度
        width: 图像宽度
        seed: 随机种子
        device: 设备
        offload: 是否使用offload
    
    Returns:
        校准数据字典
    """
    torch_device = torch.device(device)
    
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🔄 Loading Flux.1 model components with offload={'enabled' if offload else 'disabled'}...")
    
    # 初始化所有组件 - 模仿cli.py的策略
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(model_name, device="cpu" if offload else torch_device)
    ae = load_ae(model_name, device="cpu" if offload else torch_device)
    
    print(f"✓ Model components loaded")
    print(f"  T5: {sum(p.numel() for p in t5.parameters()) / 1e9:.2f}B params")
    print(f"  CLIP: {sum(p.numel() for p in clip.parameters()) / 1e9:.2f}B params") 
    print(f"  Flux: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    print(f"  VAE: {sum(p.numel() for p in ae.parameters()) / 1e9:.2f}B params")
    
    # 获取调度器
    img_seq_len = (height // 16) * (width // 16) 
    timesteps = get_schedule(num_steps, img_seq_len, shift=(model_name != "flux-schnell"))
    
    all_trajectories = {}
    all_outputs = {}
    all_txt_embeddings = []
    all_vec_embeddings = []
    all_img_ids = []
    all_txt_ids = []
    
    print(f"🎯 Generating calibration data for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        print(f"\n📝 Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # 准备输入 - 模仿cli.py的顺序
        x = get_noise(
            1,
            height,
            width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=seed + i
        )
        
        # Offload策略：先将VAE移到CPU，T5和CLIP移到GPU
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        
        # 准备输入嵌入
        inp = prepare(t5, clip, x, prompt=prompt)
        
        # 保存嵌入到CPU
        all_txt_embeddings.append(inp["txt"].cpu())
        all_vec_embeddings.append(inp["vec"].cpu()) 
        all_img_ids.append(inp["img_ids"].cpu())
        all_txt_ids.append(inp["txt_ids"].cpu())
        
        # Offload策略：将T5和CLIP移到CPU，模型移到GPU
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
        
        # 生成中间状态
        states = save_intermediate_states_with_offload(
            model=model,
            img=inp["img"],
            img_ids=inp["img_ids"],
            txt=inp["txt"],
            txt_ids=inp["txt_ids"], 
            vec=inp["vec"],
            timesteps=timesteps,
            guidance=guidance,
            offload=offload,
            device=device
        )
        
        # 保存轨迹数据（已经在CPU上）
        for t, img_state in states['trajectory'].items():
            if t not in all_trajectories:
                all_trajectories[t] = []
            all_trajectories[t].append(img_state)
        
        # 保存输出数据（已经在CPU上）
        for t, output_state in states['outputs'].items():
            if t not in all_outputs:
                all_outputs[t] = []
            all_outputs[t].append(output_state)
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ Processed prompt {i+1}, GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else "✓ Processed prompt")
    
    print("\n🔧 Assembling calibration data...")
    
    # 整理数据格式
    timesteps_tensor = torch.tensor(timesteps[1:])  # 排除初始时间步
    
    # 拼接轨迹数据 [num_timesteps, batch_size, ...]
    xs_list = []
    outputs_list = []
    
    for t in timesteps[1:]:  # 排除初始噪声
        if t in all_trajectories:
            xs_list.append(torch.stack(all_trajectories[t], dim=0))
        if t in all_outputs:
            outputs_list.append(torch.stack(all_outputs[t], dim=0))
    
    # 转换为 [batch_size, num_timesteps, ...]
    xs = torch.stack(xs_list, dim=1)  # [batch_size, num_timesteps, ...]
    outputs = torch.stack(outputs_list, dim=1)
    
    # 拼接嵌入数据
    txt_embs = torch.stack(all_txt_embeddings, dim=0)  # [batch_size, seq_len, embed_dim]
    vec_embs = torch.stack(all_vec_embeddings, dim=0)  # [batch_size, embed_dim]
    img_ids = torch.stack(all_img_ids, dim=0)  # [batch_size, seq_len, 3]
    txt_ids = torch.stack(all_txt_ids, dim=0)  # [batch_size, seq_len, 3]
    
    # 扩展时间步和嵌入到所有时间步
    ts = timesteps_tensor.unsqueeze(1).repeat(1, len(prompts))  # [num_timesteps, batch_size]
    
    # 为每个时间步复制嵌入
    num_timesteps = len(timesteps) - 1
    
    # 检查并修正张量维度
    print(f"  📊 Tensor shapes before expansion:")
    print(f"    txt_embs: {txt_embs.shape}")
    print(f"    vec_embs: {vec_embs.shape}")
    print(f"    img_ids: {img_ids.shape}")
    print(f"    txt_ids: {txt_ids.shape}")
    print(f"    num_timesteps: {num_timesteps}")
    
    # 为每个时间步复制嵌入，确保维度正确
    if len(txt_embs.shape) == 3:  # [batch_size, seq_len, embed_dim]
        txt_embs_expanded = txt_embs.unsqueeze(0).repeat(num_timesteps, 1, 1, 1)  # [num_timesteps, batch_size, seq_len, embed_dim]
    else:
        # 如果已经是4维，直接处理
        txt_embs_expanded = txt_embs
    
    if len(vec_embs.shape) == 2:  # [batch_size, embed_dim]
        vec_embs_expanded = vec_embs.unsqueeze(0).repeat(num_timesteps, 1, 1)  # [num_timesteps, batch_size, embed_dim]
    else:
        vec_embs_expanded = vec_embs
    
    if len(img_ids.shape) == 3:  # [batch_size, seq_len, 3]
        img_ids_expanded = img_ids.unsqueeze(0).repeat(num_timesteps, 1, 1, 1)  # [num_timesteps, batch_size, seq_len, 3]
    else:
        img_ids_expanded = img_ids
    
    if len(txt_ids.shape) == 3:  # [batch_size, seq_len, 3]
        txt_ids_expanded = txt_ids.unsqueeze(0).repeat(num_timesteps, 1, 1, 1)  # [num_timesteps, batch_size, seq_len, 3]
    else:
        txt_ids_expanded = txt_ids
    
    calibration_data = {
        "prompts": prompts,
        "ts": ts,  # [num_timesteps, batch_size]
        "xs": xs.transpose(0, 1),  # [num_timesteps, batch_size, ...]
        "outputs": outputs.transpose(0, 1),  # [num_timesteps, batch_size, ...]
        "txt_embs": txt_embs_expanded,  # [num_timesteps, batch_size, seq_len, embed_dim]
        "vec_embs": vec_embs_expanded,  # [num_timesteps, batch_size, embed_dim]
        "img_ids": img_ids_expanded,  # [num_timesteps, batch_size, seq_len, 3]
        "txt_ids": txt_ids_expanded,  # [num_timesteps, batch_size, seq_len, 3]
        "timesteps": timesteps,
        "guidance": guidance,
        "height": height,
        "width": width,
        "num_steps": num_steps,
        "model_name": model_name
    }
    
    print("✓ Calibration data assembled successfully!")
    
    return calibration_data


def main():
    parser = argparse.ArgumentParser(description="Generate calibration data for Flux.1 dev model")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-dev",
        help="Model name (flux-dev or flux-schnell)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save calibration data"
    )
    parser.add_argument(
        "--coco_annotations",
        type=str,
        default="/home/zcx/codes/data/coco/annotations/captions_val2014.json",
        help="Path to COCO annotations file"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=32,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="Guidance scale"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,  # 默认较小尺寸减少内存使用
        help="Image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,  # 默认较小尺寸减少内存使用
        help="Image width"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--no_offload",
        action="store_true",
        help="Disable model offloading (requires more GPU memory)"
    )
    parser.add_argument(
        "--save_debug_images",
        type=str,
        default=None,
        help="Directory to save debug images"
    )
    
    args = parser.parse_args()
    
    # 检查模型配置
    if args.model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {args.model_name}, chose from {available}")
    
    # 调整图像尺寸为16的倍数
    args.height = 16 * (args.height // 16)
    args.width = 16 * (args.width // 16)
    
    offload = not args.no_offload
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 检查输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Flux.1 calibration data generation...")
    print(f"📋 Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Offload: {'enabled' if offload else 'disabled'}")
    print(f"  Device: {args.device}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 准备COCO数据
    print("\n📚 Preparing COCO calibration data...")
    try:
        prompts, image_paths = prepare_coco_text_and_image(
            args.coco_annotations, 
            max_samples=args.n_samples
        )
        print(f"✓ Loaded {len(prompts)} prompts from COCO dataset")
        
        # 打印一些示例提示
        print("📝 Sample prompts:")
        for i, prompt in enumerate(prompts[:3]):
            print(f"  {i+1}: {prompt}")
        
    except Exception as e:
        print(f"✗ Error loading COCO data: {e}")
        return
    
    # 生成校准数据
    print("\n🎯 Generating calibration data...")
    try:
        calibration_data = generate_flux_calibration_data_with_offload(
            prompts=prompts,
            model_name=args.model_name,
            num_steps=args.num_steps,
            guidance=args.guidance,
            height=args.height,
            width=args.width,
            seed=args.seed,
            device=args.device,
            offload=offload
        )
        
        print("\n✅ Successfully generated calibration data")
        print(f"📊 Data summary:")
        for key, value in calibration_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list) and key == "prompts":
                print(f"  {key}: {len(value)} prompts")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"✗ Error generating calibration data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存校准数据
    print(f"\n💾 Saving calibration data to {args.output_path}...")
    try:
        torch.save(calibration_data, args.output_path)
        
        # 计算文件大小
        file_size = os.path.getsize(args.output_path) / 1e9
        print(f"✓ Successfully saved calibration data ({file_size:.2f}GB)")
        
        # 验证保存的数据
        loaded_data = torch.load(args.output_path, map_location='cpu')
        print(f"✓ Verification: loaded data contains {len(loaded_data)} keys")
        
    except Exception as e:
        print(f"✗ Error saving calibration data: {e}")
        return
    
    # 可选：保存调试图像
    if args.save_debug_images:
        print(f"\n🖼️ Saving debug images to {args.save_debug_images}...")
        os.makedirs(args.save_debug_images, exist_ok=True)
        
        try:
            # 重新加载VAE进行解码
            ae = load_ae(args.model_name, device=args.device)
            
            # 生成几张示例图像
            with torch.no_grad():
                for i, prompt in enumerate(prompts[:3]):
                    # 获取最终去噪结果
                    final_latent = calibration_data["xs"][-1, i:i+1].to(args.device)  # 取最后一个时间步
                    
                    # 解码为图像
                    final_latent = unpack(final_latent, args.height, args.width)
                    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                        decoded = ae.decode(final_latent)
                    
                    # 转换并保存
                    from PIL import Image
                    decoded = torch.clamp(decoded, -1, 1)
                    decoded = (decoded + 1) * 127.5
                    decoded = decoded.cpu().numpy().astype(np.uint8)
                    decoded = decoded[0].transpose(1, 2, 0)  # CHW -> HWC
                    
                    img = Image.fromarray(decoded)
                    img.save(os.path.join(args.save_debug_images, f"debug_{i:03d}.png"))
                    
            print(f"✓ Saved {min(3, len(prompts))} debug images")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save debug images: {e}")
    
    print("\n🎉 Calibration data generation completed successfully!")
    print(f"📈 Memory usage peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB" if torch.cuda.is_available() else "")
    print("\n📝 Next steps:")
    print("  1. Verify the calibration data quality")
    print("  2. Implement PTQ (Post-Training Quantization) for Flux.1")
    print("  3. Run sensitivity analysis")
    print("  4. Generate mixed precision configuration")


if __name__ == "__main__":
    main() 