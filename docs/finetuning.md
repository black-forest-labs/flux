# LoRA Fine-tuning

FLUX models support fine-tuning through Low-Rank Adaptation (LoRA), which allows efficient adaptation of the model to specific styles or domains with minimal resource requirements. LoRA is a parameter-efficient fine-tuning technique that adds small, trainable rank decomposition matrices to the existing model weights, significantly reducing memory requirements and training time.

This guide covers two approaches to LoRA fine-tuning:
1. **Native Flux Implementation** - Using Flux's built-in LoRA support
2. **Kohya_ss Scripts** - Using the popular community scripts for more customization options

## How LoRA Works in Flux

The Flux implementation of LoRA follows standard best practices:

1. Low-rank matrices are applied to all Linear layers in the model
2. The original pre-trained weights remain frozen
3. Only the LoRA parameters are updated during training
4. A scale parameter controls the contribution of the LoRA weights

The implementation can be found in `src/flux/modules/lora.py` and is wrapped around the Flux model via the `FluxLoraWrapper` class in `src/flux/model.py`.

## Available LoRA Models

We provide several pre-trained LoRA adapters that can be used directly with FLUX.1 [dev] model:

| Name | HuggingFace repo | License | sha256sum |
| ---- | ---------------- | ------- | --------- |
| `FLUX.1 Canny [dev] LoRA` | https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | 8eaa21b9c43d5e7242844deb64b8cf22ae9010f813f955ca8c05f240b8a98f7e |
| `FLUX.1 Depth [dev] LoRA` | https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | 1938b38ea0fdd98080fa3e48beb2bedfbc7ad102d8b65e6614de704a46d8b907 |

---

# Training Your Own LoRA

You can train your own LoRA models for Flux using either the native implementation or Kohya_ss scripts. Both approaches have their advantages:

- **Native Flux Implementation**: Optimized specifically for Flux, simpler integration
- **Kohya_ss Scripts**: More customization options, extensive community support, advanced features

## Approach 1: Native Flux Implementation

### Requirements

- FLUX.1 [dev] base model
- Custom training dataset (images and captions)
- At least 16GB of VRAM

### Setup

1. Ensure you have the FLUX Python library installed with training dependencies:

```bash
pip install -e ".[train]"
```

2. Download and set up the FLUX.1 [dev] base model:

```bash
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

### Preparing Training Data

Prepare your training data as a folder containing image-caption pairs. Each image should have a corresponding text file with the same name but a `.txt` extension:

```
training_data/
   image1.jpg
   image1.txt
   image2.jpg
   image2.txt
  ...
```

Text files should contain a single line with the prompt that accurately describes the image.

### Configuration

Create a training configuration file `lora_config.yaml`:

```yaml
model:
  base_model: "flux-dev"
  lora_rank: 64
  lora_scale: 1.0

training:
  batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 1e-4
  epochs: 5
  save_every: 500
  validation_every: 100
  resolution: 768
  random_crop: True

data:
  train_data_dir: "path/to/training_data"
  validation_data_dir: "path/to/validation_data"  # Optional
  validation_prompts_file: "path/to/validation_prompts.txt"  # Optional
```

### Training Process

Start the training process with:

```bash
python -m flux.training.train --config lora_config.yaml --output_dir lora_output
```

This will:
1. Initialize the FLUX.1 [dev] model
2. Apply LoRA adaptation layers according to the configuration
3. Freeze the base model weights
4. Train only the LoRA parameters
5. Save checkpoints periodically to the output directory

---

## Approach 2: Kohya_ss Scripts

[Kohya_ss scripts](https://github.com/kohya-ss/sd-scripts) are a popular community-maintained set of training scripts that offer enhanced customization and features for fine-tuning Stable Diffusion models, including Flux.

### Requirements

- FLUX.1 [dev] base model
- Custom training dataset (images and captions)
- At least 12GB of VRAM (8GB possible with optimizations)
- Python 3.10+
- CUDA compatible GPU

### Setup

1. Clone the Kohya_ss repository:

```bash
git clone https://github.com/kohya-ss/sd-scripts
cd sd-scripts
```

2. Set up a Python environment:

```bash
# Using conda (recommended)
conda create -n flux-kohya python=3.10
conda activate flux-kohya

# Or using venv
python -m venv flux-kohya
source flux-kohya/bin/activate  # On Windows: flux-kohya\Scripts\activate
```

3. Install dependencies:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -U xformers
```

### Preparing Your Dataset

1. Create a folder structure for your images and captions:

```
training_data/
├── img1.png
├── img1.txt
├── img2.jpg
├── img2.txt
└── ...
```

2. (Optional) Use metadata tagging tools to help create caption files:

```bash
# For BLIP auto-captioning (if needed)
python finetune/make_captions.py \
  --batch_size 8 \
  --caption_extension .txt \
  --caption_weights "path_to_caption_weights" \
  training_data/
```

3. Prepare buckets and latents for efficient training:

```bash
python prepare_buckets_latents.py \
  --train_data_dir training_data \
  --resolution 768,768 \
  --min_bucket_reso 256 \
  --max_bucket_reso 1024 \
  --batch_size 4 \
  --mixed_precision bf16 \
  --model_path "path/to/flux/model"
```

### Creating a Training Configuration

Create a `config.toml` file tailored for Flux:

```toml
[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = 0
caption_extension = ".txt"
mixed_precision = "bf16"

[model]
# Point to your local Flux model or use Diffusers format path
pretrained_model_name_or_path = "path/to/flux/model"
v2 = false
v_parameterization = false

[network]
type = "lora"
dim = 64  # LoRA rank (64-128 recommended for Flux)
alpha = 64  # LoRA alpha, usually same as dim
conv_dim = 32  # Rank for convolutional layers
conv_alpha = 32  # Alpha for convolutional layers

[optimizer]
type = "AdamW8bit"
learning_rate = 1e-4
max_grad_norm = 1.0
weight_decay = 0.01
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 3
lr_warmup_steps = 100

[training]
output_dir = "./flux_lora_output"
output_name = "flux-lora-model"
train_batch_size = 2  # Adjust based on VRAM
max_train_steps = 5000
max_data_loader_n_workers = 8
gradient_checkpointing = true
gradient_accumulation_steps = 2  # To achieve effective batch size of 4
save_every_n_steps = 500
save_last_n_steps = 4  # Keep the last 4 saved steps
```

### Starting the Training Process

Launch training with accelerate:

```bash
accelerate launch --num_cpu_threads_per_process 8 train_network.py \
  --train_data_dir training_data \
  --output_dir flux_lora_output \
  --pretrained_model_name_or_path "path/to/flux/model" \
  --resolution 768 \
  --network_module networks.lora \
  --network_dim 64 \
  --network_alpha 64 \
  --max_train_epochs 5 \
  --learning_rate 1e-4 \
  --unet_lr 1e-4 \
  --text_encoder_lr 5e-5 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --mixed_precision bf16 \
  --save_every_n_epochs 1 \
  --save_precision bf16 \
  --seed 42 \
  --cache_latents \
  --clip_skip 2 \
  --prior_loss_weight 1.0 \
  --max_token_length 225 \
  --xformers \
  --bucket_reso_steps 64 \
  --noise_offset 0.1
```

Alternatively, use the config file approach:

```bash
accelerate launch --num_cpu_threads_per_process 8 train_network.py \
  --train_data_dir training_data \
  --config_file config.toml
```

### Training with Text Encoder LoRA (Advanced)

For more powerful fine-tuning, train LoRA weights for both UNet and text encoder:

```bash
accelerate launch --num_cpu_threads_per_process 8 train_network.py \
  --train_data_dir training_data \
  --config_file config.toml \
  --train_text_encoder
```

### Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=flux_lora_output/tensorboard
```

---

# Using Fine-tuned LoRA Models

Once you've trained your LoRA model using either approach, you can use it with your Flux model.

## Using Native Flux LoRA Models

### Loading a LoRA Model

After training with the native approach:

```bash
# Set environment variables
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
export MY_CUSTOM_LORA=<path_to_lora_checkpoint>

# Interactive sampling
python -m flux.cli --name flux-dev --lora_path $MY_CUSTOM_LORA --lora_scale 1.0 --loop
```

### Diffusers Integration

You can also use your fine-tuned LoRA with the Diffusers library:

```python
import torch
from diffusers import FluxPipeline

# Load base model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Load and apply LoRA weights
pipe.load_lora_weights("path/to/lora_checkpoint", scale=1.0)

# Generate image with LoRA adaptation
prompt = "A portrait in the style of my fine-tuned model"
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=10,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]

image.save("lora_sample.png")
```

## Using Kohya_ss LoRA Models

Kohya_ss LoRA models are saved in Safetensors format and can be used with various tools.

### Loading with Flux CLI

```bash
# Set environment variables
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
export KOHYA_LORA=<path_to_kohya_lora>.safetensors

# Interactive sampling
python -m flux.cli --name flux-dev --lora_path $KOHYA_LORA --lora_scale 1.0 --loop
```

### Loading with Diffusers

```python
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file

# Load base model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Load Kohya LoRA weights
pipe.load_lora_weights("path/to/kohya_lora.safetensors", adapter_name="kohya_lora")
pipe.set_adapters(["kohya_lora"], adapter_weights=[1.0])

# Generate image with LoRA adaptation
prompt = "A portrait in the style of my fine-tuned model"
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=10,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]

image.save("kohya_lora_sample.png")
```

### Converting Between Formats

If needed, you can convert between Kohya_ss and native Flux LoRA formats:

```bash
# From Kohya to Flux format
python -m flux.tools.convert_lora --input path/to/kohya_lora.safetensors --output path/to/flux_lora

# From Flux to Kohya format
python -m flux.tools.convert_lora --input path/to/flux_lora --output path/to/kohya_lora.safetensors --format safetensors
```

## Best Practices for LoRA Fine-tuning

1. **Dataset Quality**: Curate a high-quality dataset with consistent style or content
2. **Dataset Size**: 20-100 images can be sufficient for style adaptation
3. **Prompt Engineering**: Ensure captions accurately describe the images and include any stylistic elements
4. **Resolution**: Train at the resolution you plan to generate at
5. **Regularization**: Consider adding some examples from the original training set to prevent overfitting
6. **LoRA Scale**: Adjust the LoRA scale at inference time to control adaptation strength
7. **LoRA Rank**: For Flux models, ranks between 64-128 tend to work well
8. **Training Duration**: Monitor validation images and stop when quality no longer improves

## Troubleshooting

### Out of Memory Errors
- Reduce batch size or use gradient accumulation
- Lower LoRA rank (dim and alpha values)
- Enable gradient checkpointing
- Use 8-bit optimizers (like AdamW8bit)
- Use mixed precision training (bf16/fp16)

### Poor Generation Results
- Check dataset quality and captions
- Extend training duration
- Adjust learning rate
- Try training text encoder LoRA as well

### Training Divergence
- Lower learning rate
- Increase warmup steps
- Implement gradient clipping

## TensorRT Engine Inference

LoRA models can also be used with TensorRT engine inference for faster generation. You'll need to export your custom LoRA-adapted model to ONNX format first:

```bash
python -m flux.export.export_onnx --model_path <path_to_base_model> --lora_path <path_to_lora_checkpoint> --lora_scale 1.0 --output_dir <onnx_export_dir>
```

Then use it for inference:

```bash
TRT_ENGINE_DIR=<your_trt_engine_dir> ONNX_DIR=<onnx_export_dir> python src/flux/cli.py --prompt "<prompt>" --trt --static_shape=False --name=flux-dev --trt_transformer_precision bf16
```

## References

- [Kohya_ss GitHub Repository](https://github.com/kohya-ss/sd-scripts)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Flux GitHub Repository](https://github.com/black-forest-labs/flux)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/index)