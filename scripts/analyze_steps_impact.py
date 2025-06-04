#!/usr/bin/env python3
"""
分析不同推理步数对Flux.1 dev校准数据质量的影响

基于MixDQ的思路：
1. 校准数据的目标是捕获模型激活的统计分布
2. 不同推理步数下的激活分布覆盖度分析
3. 计算成本 vs 校准质量的权衡分析
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import argparse
from pathlib import Path

def analyze_activation_coverage(data_dict: Dict, step_counts: List[int]) -> Dict:
    """分析不同步数下的激活覆盖度"""
    
    results = {}
    
    for steps in step_counts:
        if f"flux_calib_{steps}_steps.pt" in data_dict:
            calib_data = data_dict[f"flux_calib_{steps}_steps.pt"]
            
            # 分析激活张量的统计分布
            xs = calib_data.get("xs", None)
            if xs is not None:
                # 计算激活值的统计指标
                mean_vals = xs.mean(dim=(0, 2, 3, 4))  # 跨batch和空间维度
                std_vals = xs.std(dim=(0, 2, 3, 4))
                min_vals = xs.min(dim=0)[0].flatten()
                max_vals = xs.max(dim=0)[0].flatten()
                
                # 计算分布熵（作为多样性指标）
                hist_data = []
                for t in range(xs.size(0)):  # 跨时间步
                    flat_vals = xs[t].flatten()
                    hist, _ = np.histogram(flat_vals.cpu().numpy(), bins=50, density=True)
                    hist = hist + 1e-10  # 避免log(0)
                    entropy = -np.sum(hist * np.log(hist))
                    hist_data.append(entropy)
                
                results[steps] = {
                    'mean_coverage': mean_vals.std().item(),  # 均值的变化范围
                    'std_coverage': std_vals.mean().item(),   # 标准差的平均值
                    'range_coverage': (max_vals.max() - min_vals.min()).item(),
                    'entropy_mean': np.mean(hist_data),
                    'entropy_std': np.std(hist_data),
                    'num_timesteps': xs.size(0),
                    'activation_shape': xs.shape
                }
    
    return results

def generate_test_data_different_steps(steps_list: List[int], output_dir: str) -> None:
    """生成不同步数的测试校准数据"""
    
    print(f"🧪 生成不同步数的测试数据...")
    
    for steps in steps_list:
        output_path = os.path.join(output_dir, f"flux_calib_{steps}_steps.pt")
        
        if os.path.exists(output_path):
            print(f"✓ 已存在：{steps}步数据")
            continue
            
        print(f"📊 生成{steps}步校准数据...")
        
        cmd = f"""python scripts/gen_flux_calib_data.py \
            --output_path {output_path} \
            --n_samples 8 \
            --height 256 \
            --width 256 \
            --num_steps {steps} \
            --seed 42"""
        
        os.system(cmd)

def plot_steps_analysis(results: Dict, output_dir: str) -> None:
    """绘制步数分析结果"""
    
    steps = sorted(results.keys())
    
    # 提取指标
    coverage_metrics = {
        'mean_coverage': [results[s]['mean_coverage'] for s in steps],
        'std_coverage': [results[s]['std_coverage'] for s in steps],
        'range_coverage': [results[s]['range_coverage'] for s in steps],
        'entropy_mean': [results[s]['entropy_mean'] for s in steps]
    }
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Flux.1 Dev: 不同推理步数对校准数据质量的影响', fontsize=16)
    
    # 1. 激活均值覆盖度
    axes[0, 0].plot(steps, coverage_metrics['mean_coverage'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('激活均值覆盖度')
    axes[0, 0].set_xlabel('推理步数')
    axes[0, 0].set_ylabel('均值变化范围')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=30, color='red', linestyle='--', alpha=0.7, label='推荐(30步)')
    axes[0, 0].legend()
    
    # 2. 标准差覆盖度
    axes[0, 1].plot(steps, coverage_metrics['std_coverage'], 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_title('激活标准差覆盖度')
    axes[0, 1].set_xlabel('推理步数')
    axes[0, 1].set_ylabel('标准差平均值')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=30, color='red', linestyle='--', alpha=0.7, label='推荐(30步)')
    axes[0, 1].legend()
    
    # 3. 激活范围覆盖度
    axes[1, 0].plot(steps, coverage_metrics['range_coverage'], 'mo-', linewidth=2, markersize=8)
    axes[1, 0].set_title('激活值范围覆盖度')
    axes[1, 0].set_xlabel('推理步数')
    axes[1, 0].set_ylabel('最大-最小值范围')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=30, color='red', linestyle='--', alpha=0.7, label='推荐(30步)')
    axes[1, 0].legend()
    
    # 4. 分布熵（多样性）
    axes[1, 1].plot(steps, coverage_metrics['entropy_mean'], 'co-', linewidth=2, markersize=8)
    axes[1, 1].set_title('激活分布多样性 (熵)')
    axes[1, 1].set_xlabel('推理步数')
    axes[1, 1].set_ylabel('平均熵值')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=30, color='red', linestyle='--', alpha=0.7, label='推荐(30步)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steps_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def calculate_efficiency_ratio(results: Dict) -> Dict:
    """计算效率比：质量提升 / 计算成本增加"""
    
    steps = sorted(results.keys())
    base_steps = min(steps)  # 以最少步数为基准
    
    efficiency = {}
    
    for i, current_steps in enumerate(steps[1:], 1):
        base_quality = results[base_steps]['entropy_mean']
        current_quality = results[current_steps]['entropy_mean']
        
        quality_improvement = (current_quality - base_quality) / base_quality
        cost_increase = (current_steps - base_steps) / base_steps
        
        efficiency_ratio = quality_improvement / cost_increase if cost_increase > 0 else 0
        
        efficiency[current_steps] = {
            'quality_improvement': quality_improvement * 100,  # 百分比
            'cost_increase': cost_increase * 100,              # 百分比  
            'efficiency_ratio': efficiency_ratio
        }
    
    return efficiency

def print_recommendations(results: Dict, efficiency: Dict) -> None:
    """打印基于分析的推荐"""
    
    print("\n" + "="*60)
    print("📊 Flux.1 Dev 推理步数分析报告")
    print("="*60)
    
    print("\n🎯 基于MixDQ思路的分析结果：")
    
    # 找到最佳效率点
    best_efficiency_steps = max(efficiency.keys(), key=lambda x: efficiency[x]['efficiency_ratio'])
    
    print(f"\n📈 不同步数的质量指标：")
    for steps in sorted(results.keys()):
        entropy = results[steps]['entropy_mean']
        coverage = results[steps]['mean_coverage']
        print(f"  {steps}步: 熵={entropy:.3f}, 覆盖度={coverage:.3f}")
    
    print(f"\n⚡ 效率分析：")
    for steps in sorted(efficiency.keys()):
        eff = efficiency[steps]
        print(f"  {steps}步: 质量提升={eff['quality_improvement']:+.1f}%, "
              f"成本增加={eff['cost_increase']:+.1f}%, 效率比={eff['efficiency_ratio']:.3f}")
    
    print(f"\n🎯 推荐策略：")
    print(f"  💡 最佳效率点：{best_efficiency_steps}步")
    print(f"  🔬 快速开发：4-10步")
    print(f"  🏭 生产应用：20-30步")
    print(f"  🔧 高质量：40-50步")
    
    print(f"\n📝 MixDQ对比分析：")
    print(f"  • SDXL (50步模型) → 使用30步校准 (60%)")
    print(f"  • Flux.1 dev (50步模型) → 推荐30步校准 (60%)")
    print(f"  • 策略一致性：两者都采用约60%的完整步数")
    
    print(f"\n💡 实用建议：")
    print(f"  • 30步能够覆盖大部分重要的去噪阶段")
    print(f"  • 校准重点是激活统计分布，不是图像质量")
    print(f"  • 可以从20步开始，根据量化结果调整")
    print(f"  • 完整50步仅在最终优化时考虑")

def main():
    parser = argparse.ArgumentParser(description="分析Flux.1 dev不同推理步数的影响")
    parser.add_argument("--output_dir", type=str, default="./steps_analysis", 
                       help="输出目录")
    parser.add_argument("--generate_data", action="store_true",
                       help="是否生成测试数据")
    parser.add_argument("--steps_list", nargs='+', type=int, 
                       default=[4, 10, 20, 30, 40, 50],
                       help="要分析的步数列表")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成测试数据（如果需要）
    if args.generate_data:
        generate_test_data_different_steps(args.steps_list, args.output_dir)
    
    # 加载现有数据进行分析
    data_dict = {}
    for steps in args.steps_list:
        filepath = os.path.join(args.output_dir, f"flux_calib_{steps}_steps.pt")
        if os.path.exists(filepath):
            print(f"📂 加载{steps}步数据...")
            data_dict[f"flux_calib_{steps}_steps.pt"] = torch.load(filepath, map_location='cpu')
    
    if not data_dict:
        print("❌ 未找到校准数据文件，请先使用 --generate_data 生成")
        return
    
    # 分析激活覆盖度
    print("\n🔍 分析激活覆盖度...")
    results = analyze_activation_coverage(data_dict, args.steps_list)
    
    # 计算效率比
    print("📊 计算效率比...")
    efficiency = calculate_efficiency_ratio(results)
    
    # 绘制分析图
    print("📈 生成分析图表...")
    plot_steps_analysis(results, args.output_dir)
    
    # 打印推荐
    print_recommendations(results, efficiency)
    
    print(f"\n✅ 分析完成！结果保存在：{args.output_dir}")

if __name__ == "__main__":
    main() 