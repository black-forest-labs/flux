from flux.modules.autoencoder import AutoEncoder
from flux.modules.conditioner import HFEmbedder
from flux.model import Flux


class TRTBuilder:
    __stages__ = ["clip", "t5", "transformer", "vae"]

    @property
    def stages(self) -> list[str]:
        return self.__stages__

    def __init__(
        self,
        flux_model: Flux,
        t5_model: HFEmbedder,
        clip_model: HFEmbedder,
        ae_model: AutoEncoder,
    ):
        self.models = {
            "clip": clip_model,
            "transformer": flux_model,
            "t5": t5_model,
            "ae": ae_model,
        }


# def _prepare_model_configs(self, onnx_dir, engine_dir, enable_refit, int8, fp8, quantization_level, quantization_percentile, quantization_alpha, calibration_size):
#     model_names = self.models.keys()
#     lora_suffix = self._get_lora_suffix()
#     self.torch_fallback = dict(zip(model_names, [self.torch_inference or self.config.get(model_name.replace('-','_')+'_torch_fallback', False) for model_name in model_names]))

#     configs = {}
#     for model_name in model_names:
#         config = {
#             'do_engine_refit': not self.pipeline_type.is_sd_xl_refiner() and enable_refit and model_name.startswith('unet'),
#             'do_lora_merge': not enable_refit and self.lora_loader and model_name.startswith('unet'),
#             'use_int8': False,
#             'use_fp8': False,
#         }
#         config['model_suffix'] = lora_suffix if config['do_lora_merge'] else ''

#         if int8:
#             assert self.pipeline_type.is_sd_xl_base() or self.version in ["1.5", "2.1", "2.1-base"], "int8 quantization only supported for SDXL, SD1.5 and SD2.1 pipeline"
#             if model_name == ('unetxl' if self.pipeline_type.is_sd_xl() else 'unet'):
#                 config['use_int8'] = True
#                 config['model_suffix'] += f"-int8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}.p{quantization_percentile}.a{quantization_alpha}"
#         elif fp8:
#             assert self.pipeline_type.is_sd_xl() or self.version in ["1.5", "2.1", "2.1-base"], "fp8 quantization only supported for SDXL, SD1.5 and SD2.1 pipeline"
#             if model_name == ('unetxl' if self.pipeline_type.is_sd_xl() else 'unet'):
#                 config['use_fp8'] = True
#                 config['model_suffix'] += f"-fp8.l{quantization_level}.bs2.s{self.denoising_steps}.c{calibration_size}.p{quantization_percentile}.a{quantization_alpha}"

#         config['onnx_path'] = self._get_onnx_path(model_name, onnx_dir, opt=False, suffix=config['model_suffix'])
#         config['onnx_opt_path'] = self._get_onnx_path(model_name, onnx_dir, suffix=config['model_suffix'])
#         config['engine_path'] = self._get_engine_path(model_name, engine_dir, config['do_engine_refit'], suffix=config['model_suffix'])
#         config['weights_map_path'] = self._get_weights_map_path(model_name, onnx_dir) if config['do_engine_refit'] else None
#         config['state_dict_path'] = self._get_state_dict_path(model_name, onnx_dir, suffix=config['model_suffix'])
#         config['refit_weights_path'] = self._get_refit_nodes_path(model_name, onnx_dir, suffix=lora_suffix)

#         configs[model_name] = config

#     return configs


# def load_engines(
#     self,
#     engine_dir,
#     framework_model_dir,
#     onnx_dir,
#     onnx_opset,
#     opt_batch_size,
#     opt_image_height,
#     opt_image_width,
#     optimization_level=3,
#     static_batch=False,
#     static_shape=True,
#     enable_refit=False,
#     enable_all_tactics=False,
#     timing_cache=None,
#     int8=False,
#     fp8=False,
#     quantization_level=2.5,
#     quantization_percentile=1.0,
#     quantization_alpha=0.8,
#     calibration_size=32,
#     calib_batch_size=2,
# ):
