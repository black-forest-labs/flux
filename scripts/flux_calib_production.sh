#!/bin/bash

# Flux.1 Dev 校准数据生成脚本
# 基于MixDQ思路，针对Flux.1 dev的50步推理特性优化

set -e

# 配置参数 - 基于Flux.1 dev的推理特性
N_SAMPLES=64        # 标准样本数量
RESOLUTION=1024     # Flux.1 dev的标准分辨率
NUM_STEPS=30        # 关键参数：60%的完整步数，参考MixDQ的SDXL策略
BASE_DIR="flux_quantization_results"
OUTPUT_DIR="${BASE_DIR}/calibration_data"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "🚀 Flux.1 Dev 校准数据生成"
echo "========================================"
echo "📋 配置参数："
echo "  🎯 模型：Flux.1 dev (50步推理模型)"
echo "  📊 校准步数：${NUM_STEPS} (约为完整50步的60%)"
echo "  📈 样本数量：${N_SAMPLES}"
echo "  🖼️  分辨率：${RESOLUTION}x${RESOLUTION}"
echo "  📁 输出目录：${OUTPUT_DIR}"
echo ""

# 设置环境变量
export FLUX_DEV="/home/zcx/codes/flux1dev/flux1-dev.safetensors"
export AE="/home/zcx/codes/flux1dev/ae.safetensors"

# 检查模型文件
if [[ ! -f "$FLUX_DEV" ]]; then
    echo "❌ 错误：找不到Flux dev模型文件：$FLUX_DEV"
    exit 1
fi

if [[ ! -f "$AE" ]]; then
    echo "❌ 错误：找不到VAE模型文件：$AE"
    exit 1
fi

echo "✅ 模型文件检查完成"
echo ""

# 1. 快速测试 - 验证流程
echo "1️⃣ 生成测试数据（4步，快速验证）..."
python scripts/gen_flux_calib_data.py \
    --output_path "${OUTPUT_DIR}/flux_calib_test.pt" \
    --n_samples 2 \
    --height 256 \
    --width 256 \
    --num_steps 4 \
    --save_debug_images "${OUTPUT_DIR}/debug_test"

# 2. 开发配置 - 20步，适合算法开发
echo "2️⃣ 生成开发配置数据（20步）..."
python scripts/gen_flux_calib_data.py \
    --output_path "${OUTPUT_DIR}/flux_calib_development.pt" \
    --n_samples 32 \
    --height 512 \
    --width 512 \
    --num_steps 20

# 3. 标准配置 - 30步，生产推荐
echo "3️⃣ 生成标准配置数据（30步，推荐用于生产）..."
python scripts/gen_flux_calib_data.py \
    --output_path "${OUTPUT_DIR}/flux_calib_standard.pt" \
    --n_samples $N_SAMPLES \
    --height $RESOLUTION \
    --width $RESOLUTION \
    --num_steps $NUM_STEPS

# 4. 保守配置 - 40步，高质量保证
echo "4️⃣ 生成保守配置数据（40步，高质量）..."
python scripts/gen_flux_calib_data.py \
    --output_path "${OUTPUT_DIR}/flux_calib_conservative.pt" \
    --n_samples 96 \
    --height $RESOLUTION \
    --width $RESOLUTION \
    --num_steps 40

# 5. 完整配置 - 50步（可选，资源充足时）
read -p "是否生成完整50步校准数据？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "5️⃣ 生成完整配置数据（50步，最高质量）..."
    python scripts/gen_flux_calib_data.py \
        --output_path "${OUTPUT_DIR}/flux_calib_full.pt" \
        --n_samples 128 \
        --height $RESOLUTION \
        --width $RESOLUTION \
        --num_steps 50 \
        --save_debug_images "${OUTPUT_DIR}/debug_full"
else
    echo "⏭️ 跳过完整50步配置"
fi

echo ""
echo "✅ 校准数据生成完成！"
echo "📁 生成的文件："
ls -lh "${OUTPUT_DIR}"/*.pt

echo ""
echo "📊 步数策略说明："
echo "  • 4步  - 快速测试，验证流程"
echo "  • 20步 - 开发阶段，平衡效率与质量"
echo "  • 30步 - 生产推荐，60%完整步数（参考MixDQ的SDXL策略）"
echo "  • 40步 - 保守策略，80%完整步数"
echo "  • 50步 - 完整步数，最高质量但计算成本高"

echo ""
echo "💡 推荐使用策略："
echo "  🔬 算法开发：使用 flux_calib_development.pt (20步)"
echo "  🏭 生产部署：使用 flux_calib_standard.pt (30步)"
echo "  🔧 最终优化：使用 flux_calib_conservative.pt (40步)"

echo ""
echo "📝 下一步混合精度量化："
echo "  1. 基于校准数据实施PTQ (Post-Training Quantization)"
echo "  2. 进行敏感度分析确定量化策略"
echo "  3. 生成mixed precision配置"
echo "  4. 评估量化后的模型质量"

echo ""
echo "⚡ 性能建议："
echo "  • 30步配置在大多数情况下足够用于量化校准"
echo "  • 校准数据关注激活分布统计，不需要完美的图像生成"
echo "  • 可以根据量化结果迭代调整步数" 