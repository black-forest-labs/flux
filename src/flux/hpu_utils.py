from typing import Optional

import torch

def load_model_to_hpu(model, model_name: Optional[str] = None):
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    load_habana_module()

    if not torch.hpu.is_available():
        return model

    # Adapt transformers models to Gaudi for optimization
    adapt_transformers_to_gaudi()

    if model.__class__.__name__ == "HFEmbedder" and model_name:
        if "t5" in model_name.lower():
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif "clip" in model_name.lower():
            from optimum.habana.transformers.models.clip import GaudiCLIPVisionModel
            model = GaudiCLIPVisionModel.from_pretrained(
                model_name,
                use_flash_attention=True,
                flash_attention_recompute=False,
                torch_dtype=torch.bfloat16
            )
    else:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)
        model = model.to(torch.device("hpu"), dtype=torch.bfloat16)

    return model.eval()

def get_dtype(device: str) -> torch.dtype:
    """
    Determine the appropriate dtype to use based on the device.

    Args:
        device (str): Device string ('cuda', 'hpu', or 'cpu').

    Returns:
        torch.dtype: Data type (torch.float32 or torch.bfloat16).
    """
    if "hpu" in device:
        return torch.bfloat16
    return torch.float32
