import torch


def load_model_to_hpu(model):
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    load_habana_module()

    device = "hpu"
    if torch.hpu.is_available():
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)
        model = model.eval().to(torch.device(device))
    return model


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
