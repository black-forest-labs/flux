# SAM3 + FLUX.2-dev Point-Prompted Inpainting

This module combines Meta's **SAM3** (Segment Anything Model 3) with **FLUX.2-dev** for interactive point-prompted image inpainting.

## How It Works

1. **Click** on an image to select a region (using point prompts)
2. **SAM3** generates a precise segmentation mask from your clicks
3. **FLUX.2-dev** inpaints the masked region based on your text prompt

## Installation

### Prerequisites

1. **SAM3** - Request access and install:
   ```bash
   # Request access at: https://huggingface.co/facebook/sam3

   # Install SAM3
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   ```

2. **FLUX.2-dev** - Request access:
   ```bash
   # Request access at: https://huggingface.co/black-forest-labs/FLUX.2-dev

   # Login to HuggingFace
   huggingface-cli login
   ```

3. **Install this package**:
   ```bash
   pip install -e ".[inpainting]"
   ```

### System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (24GB+ recommended)
- **CUDA**: 12.1+ recommended
- **Python**: 3.10+

For lower VRAM (<24GB), use the `low_vram=True` option to enable CPU offloading.

## Usage

### Gradio Demo (Recommended)

The easiest way to use the inpainting feature:

```bash
# Launch the interactive demo
python demo_inpaint.py

# Or via CLI
flux-inpaint demo
```

Then open http://localhost:7860 in your browser.

### Python API

```python
from flux import create_pipeline

# Create pipeline (lazy loads models)
pipeline = create_pipeline(low_vram=True)

# Inpaint with point prompts
result = pipeline.inpaint_at_points(
    image="input.jpg",
    points=[(100, 200), (150, 220)],  # Click coordinates
    labels=[1, 1],  # 1=foreground (include), 0=background (exclude)
    prompt="a golden retriever sitting",
    num_inference_steps=28,
    guidance_scale=3.5,
)
result.save("output.jpg")
```

### CLI

```bash
# Inpaint using point prompts
flux-inpaint inpaint \
    --image input.jpg \
    --points "100,200;150,220" \
    --prompt "a golden retriever" \
    --output result.png

# Generate mask only (no inpainting)
flux-inpaint segment \
    --image input.jpg \
    --points "100,200;150,220" \
    --output mask.png \
    --visualize

# Launch Gradio demo
flux-inpaint demo --port 7860
```

## API Reference

### SAMFluxPipeline

The main pipeline class combining SAM3 and FLUX.2-dev.

```python
from flux import SAMFluxPipeline

pipeline = SAMFluxPipeline(
    sam_device="cuda",        # Device for SAM3
    flux_device="cuda",       # Device for FLUX
    sam_model_size="large",   # SAM3 size: "large", "base_plus", "small"
    flux_model="black-forest-labs/FLUX.2-dev",
    low_vram=False,           # Enable CPU offloading
    lazy_load=True,           # Load models on first use
)
```

#### Methods

- **`segment(image, points, labels, dilate_pixels)`** - Generate mask from points
- **`inpaint(image, mask, prompt, ...)`** - Inpaint with existing mask
- **`inpaint_at_points(image, points, prompt, ...)`** - Full pipeline: points → mask → inpaint

### SAM3Segmenter

Standalone SAM3 wrapper for mask generation.

```python
from flux import SAM3Segmenter

segmenter = SAM3Segmenter(device="cuda", model_size="large")

# Generate mask
mask, score = segmenter.segment(
    image=image,
    points=[(x, y)],
    labels=[1],  # 1=foreground
)

# Get PIL mask (white on black)
mask_pil = segmenter.segment_to_pil(image, points, labels)
```

### Flux2Inpainter

Standalone FLUX.2-dev inpainting wrapper.

```python
from flux import Flux2Inpainter

inpainter = Flux2Inpainter(
    model_name="black-forest-labs/FLUX.2-dev",
    device="cuda",
)

result = inpainter.inpaint(
    image=image,
    mask=mask,  # White=inpaint, Black=keep
    prompt="description of what to generate",
    num_inference_steps=28,
    guidance_scale=3.5,
    strength=0.85,
)
```

## Parameters

### Segmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `points` | required | List of (x, y) click coordinates |
| `labels` | all 1s | Point labels: 1=foreground, 0=background |
| `dilate_pixels` | 5 | Expand mask edges by N pixels |

### Inpainting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | required | What to generate in masked area |
| `negative_prompt` | "" | What to avoid |
| `num_inference_steps` | 28 | More steps = better quality |
| `guidance_scale` | 3.5 | How closely to follow prompt |
| `strength` | 0.85 | How much to change (0-1) |
| `seed` | random | For reproducibility |

## Tips

### Getting Good Masks

- **Multiple points**: Click multiple times to refine selection
- **Background points**: Use label=0 to exclude areas
- **Dilation**: Increase `dilate_pixels` for softer edges

### Getting Good Inpainting Results

- **Be specific**: Describe exactly what you want
- **Match style**: Mention lighting, style if relevant
- **Adjust strength**: Lower values preserve more original content
- **Try different seeds**: Results vary with seed

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
│  (Gradio demo or Python API)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                    Click points (x, y)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SAM3Segmenter                             │
│  • facebook/sam3 model                                       │
│  • Point → binary mask                                       │
│  • ~100ms inference                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                         Binary mask
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flux2Inpainter                            │
│  • FLUX.2-dev 32B model                                      │
│  • Mask + prompt → inpainted image                           │
│  • Remote text encoder (saves VRAM)                          │
│  • ~15-30s inference @ 28 steps                              │
└─────────────────────────────────────────────────────────────┘
                              │
                        Inpainted image
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Output                                 │
│  (PIL Image / saved file)                                    │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Out of Memory

```python
# Enable CPU offloading
pipeline = create_pipeline(low_vram=True)

# Or use smaller SAM model
pipeline = create_pipeline(sam_size="small")
```

### SAM3 Import Error

Ensure SAM3 is installed:
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

### FLUX.2-dev Access Denied

Request access at https://huggingface.co/black-forest-labs/FLUX.2-dev and login:
```bash
huggingface-cli login
```

## License

This module combines:
- **SAM3**: Meta AI License
- **FLUX.2-dev**: FLUX [dev] Non-Commercial License

Please review both licenses for your use case.
