#!/usr/bin/env python3
"""
简化的校准数据生成测试脚本
用于验证Flux.1 dev模型校准数据生成的正确性
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# 添加flux模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_model_loading():
    """测试模型加载"""
    print("🔍 Testing model loading...")
    
    try:
        from flux.util import load_flow_model, load_t5, load_clip, load_ae
        
        # 测试加载Flux模型
        print("  Loading Flux model...")
        model = load_flow_model("flux-dev", device="cuda", hf_download=True)
        print(f"  ✓ Flux model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
        
        # 测试加载T5
        print("  Loading T5 encoder...")
        t5 = load_t5(device="cuda", max_length=512)
        print("  ✓ T5 encoder loaded")
        
        # 测试加载CLIP
        print("  Loading CLIP encoder...")
        clip = load_clip(device="cuda")
        print("  ✓ CLIP encoder loaded")
        
        # 测试加载VAE
        print("  Loading VAE...")
        ae = load_ae("flux-dev", device="cuda")
        print("  ✓ VAE loaded")
        
        return model, t5, clip, ae
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        traceback.print_exc()
        return None, None, None, None

def test_coco_data_loading():
    """测试COCO数据加载"""
    print("\n🔍 Testing COCO data loading...")
    
    try:
        coco_path = "/home/zcx/codes/data/coco/annotations/captions_val2014.json"
        
        if not os.path.exists(coco_path):
            print(f"  ✗ COCO annotations file not found: {coco_path}")
            return None
        
        # 导入并测试COCO数据加载函数
        sys.path.append(os.path.dirname(__file__))
        from gen_flux_calib_data import prepare_coco_text_and_image
        
        prompts, image_paths = prepare_coco_text_and_image(coco_path, max_samples=5)
        
        print(f"  ✓ Loaded {len(prompts)} sample prompts")
        print("  Sample prompts:")
        for i, prompt in enumerate(prompts):
            print(f"    {i+1}: {prompt[:60]}...")
            
        return prompts
        
    except Exception as e:
        print(f"  ✗ COCO data loading failed: {e}")
        traceback.print_exc()
        return None

def test_flux_inference():
    """测试Flux模型推理"""
    print("\n🔍 Testing Flux inference...")
    
    try:
        from flux.sampling import get_noise, prepare, get_schedule
        from flux.util import load_flow_model, load_t5, load_clip
        
        # 加载模型组件
        model = load_flow_model("flux-dev", device="cuda")
        t5 = load_t5(device="cuda", max_length=512)
        clip = load_clip(device="cuda")
        
        # 准备测试输入
        prompt = "A beautiful sunset over the ocean"
        height, width = 512, 512  # 使用较小尺寸进行测试
        
        # 获取噪声
        img = get_noise(
            num_samples=1,
            height=height,
            width=width,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            seed=42
        )
        
        # 准备输入
        inp = prepare(t5, clip, img, prompt)
        
        print(f"  ✓ Input prepared:")
        print(f"    img: {inp['img'].shape}")
        print(f"    txt: {inp['txt'].shape}")
        print(f"    vec: {inp['vec'].shape}")
        print(f"    img_ids: {inp['img_ids'].shape}")
        print(f"    txt_ids: {inp['txt_ids'].shape}")
        
        # 测试单步前向传播
        with torch.no_grad():
            output = model(
                img=inp["img"],
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                timesteps=torch.tensor([0.5], device="cuda"),
                y=inp["vec"],
                guidance=torch.tensor([0.0], device="cuda") if model.params.guidance_embed else None,
            )
        
        print(f"  ✓ Forward pass successful: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Flux inference failed: {e}")
        traceback.print_exc()
        return False

def test_calibration_data_generation():
    """测试校准数据生成"""
    print("\n🔍 Testing calibration data generation...")
    
    try:
        sys.path.append(os.path.dirname(__file__))
        from gen_flux_calib_data import generate_flux_calibration_data, prepare_coco_text_and_image
        from flux.util import load_flow_model, load_t5, load_clip, load_ae
        
        # 加载模型
        model = load_flow_model("flux-dev", device="cuda")
        t5 = load_t5(device="cuda", max_length=512)
        clip = load_clip(device="cuda")
        ae = load_ae("flux-dev", device="cuda")
        
        # 准备少量测试数据
        coco_path = "/home/zcx/codes/data/coco/annotations/captions_val2014.json"
        prompts, _ = prepare_coco_text_and_image(coco_path, max_samples=2)
        
        print(f"  Testing with {len(prompts)} prompts...")
        
        # 生成校准数据（使用较小尺寸和较少步数）
        calibration_data = generate_flux_calibration_data(
            prompts=prompts,
            model=model,
            t5=t5,
            clip=clip,
            ae=ae,
            num_steps=2,  # 较少步数
            guidance=0.0,
            height=512,   # 较小尺寸
            width=512,
            seed=42,
            device="cuda"
        )
        
        print("  ✓ Calibration data generated successfully!")
        print(f"  Data structure:")
        for key, value in calibration_data.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list) and key == "prompts":
                print(f"    {key}: {len(value)} prompts")
            else:
                print(f"    {key}: {value}")
        
        # 验证数据维度
        assert calibration_data["xs"].shape[1] == len(prompts), "Batch size mismatch in xs"
        assert calibration_data["ts"].shape[1] == len(prompts), "Batch size mismatch in ts"
        assert len(calibration_data["prompts"]) == len(prompts), "Prompts count mismatch"
        
        print("  ✓ Data validation passed!")
        
        return calibration_data
        
    except Exception as e:
        print(f"  ✗ Calibration data generation failed: {e}")
        traceback.print_exc()
        return None

def test_data_save_load():
    """测试数据保存和加载"""
    print("\n🔍 Testing data save/load...")
    
    try:
        # 创建测试数据
        test_data = {
            "prompts": ["test prompt 1", "test prompt 2"],
            "xs": torch.randn(2, 2, 16, 32, 32),
            "ts": torch.tensor([[0.5, 0.0], [0.5, 0.0]]),
            "txt_embs": torch.randn(2, 2, 77, 4096),
        }
        
        # 保存测试数据
        test_path = "/tmp/test_flux_calib.pt"
        torch.save(test_data, test_path)
        print("  ✓ Test data saved")
        
        # 加载并验证
        loaded_data = torch.load(test_path, map_location="cpu")
        
        for key in test_data.keys():
            if isinstance(test_data[key], torch.Tensor):
                assert torch.equal(test_data[key], loaded_data[key]), f"Tensor mismatch for {key}"
            else:
                assert test_data[key] == loaded_data[key], f"Value mismatch for {key}"
        
        print("  ✓ Data save/load validation passed!")
        
        # 清理
        os.remove(test_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data save/load failed: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Starting Flux.1 calibration data generation tests...\n")
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPU.")
        return
    
    print(f"💾 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
    
    # 运行各项测试
    success_count = 0
    total_tests = 5
    
    # 1. 测试模型加载
    model, t5, clip, ae = test_model_loading()
    if model is not None:
        success_count += 1
    
    # 2. 测试COCO数据加载
    prompts = test_coco_data_loading()
    if prompts is not None:
        success_count += 1
    
    # 3. 测试Flux推理
    if test_flux_inference():
        success_count += 1
    
    # 4. 测试校准数据生成
    if test_calibration_data_generation():
        success_count += 1
    
    # 5. 测试数据保存加载
    if test_data_save_load():
        success_count += 1
    
    # 输出结果
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The calibration data generation system is working correctly.")
        print("\n📝 Next steps:")
        print("  1. Run the full calibration data generation:")
        print("     python scripts/gen_flux_calib_data.py --output_path flux_calib_data.pt")
        print("  2. Verify the generated data size and quality")
        print("  3. Proceed with quantization implementation")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("   Common issues:")
        print("   - Model weights not downloaded (run with hf_download=True)")
        print("   - Insufficient GPU memory (try smaller batch size)")
        print("   - Missing dependencies (check requirements)")

if __name__ == "__main__":
    main() 