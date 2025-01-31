from typing import Optional

import torch


def load_model_to_hpu(model, model_name=Optional[str]):
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    load_habana_module()

    if not torch.hpu.is_available():
        return model

    # Check if model is HFEmbedder (which wraps CLIP or T5)
    if model.__class__.__name__ == "HFEmbedder" and model_name:
        if "t5" in model_name.lower():
            from transformers.models.t5.modeling_t5 import (
                T5ForConditionalGeneration
            )
            from optimum.habana.transformers.models.t5.modeling_t5 import (
                gaudi_T5ForConditionalGeneration_forward,
                gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation
            )

            # Apply HPU-specific optimizations
            T5ForConditionalGeneration.forward = gaudi_T5ForConditionalGeneration_forward
            T5ForConditionalGeneration.prepare_inputs_for_generation = gaudi_T5ForConditionalGeneration_prepare_inputs_for_generation

            # Initialize optimized model
            model = T5ForConditionalGeneration.from_pretrained(model_name)

        elif "clip" in model_name.lower():
            from optimum.habana.transformers.models.clip import GaudiCLIPVisionModel
            model = GaudiCLIPVisionModel.from_pretrained(
                model_name,
                use_flash_attention=True,
                flash_attention_recompute=False
            )

    # Fallback to regular HPU loading
    else:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)
        model = model.to(torch.device("hpu"))

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
        return torch.float32
    return torch.bfloat16
