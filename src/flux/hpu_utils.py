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
