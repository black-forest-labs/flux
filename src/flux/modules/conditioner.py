from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
import os


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        
        # 更准确地识别CLIP模型
        if os.path.exists(version) and os.path.isdir(version):
            # 对于本地路径，检查config.json来确定模型类型
            config_path = os.path.join(version, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.is_clip = config.get("model_type") == "clip_text_model"
            else:
                # 回退到路径名判断
                self.is_clip = "text_encoder" in version and "text_encoder_2" not in version
        else:
            # 对于远程模型，使用原有逻辑
            self.is_clip = version.startswith("openai") or "clip" in version.lower()
        
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # 处理本地路径的情况
        if os.path.exists(version) and os.path.isdir(version):
            # 这是本地路径
            model_path = version
            
            if self.is_clip:
                # 对于CLIP，尝试找到对应的tokenizer路径
                tokenizer_path = self._find_tokenizer_path(model_path, "clip")
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=max_length)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(model_path, **hf_kwargs)
            else:
                # 对于T5，尝试找到对应的tokenizer路径
                tokenizer_path = self._find_tokenizer_path(model_path, "t5")
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, max_length=max_length)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(model_path, **hf_kwargs)
        else:
            # 这是从 Hugging Face Hub 加载的情况
            if self.is_clip:
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
            else:
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def _find_tokenizer_path(self, model_path: str, model_type: str) -> str:
        """
        根据模型路径找到对应的tokenizer路径
        """
        # 获取父目录
        parent_dir = os.path.dirname(model_path)
        
        if model_type == "clip":
            # 对于CLIP，寻找tokenizer文件夹
            tokenizer_path = os.path.join(parent_dir, "tokenizer")
        else:
            # 对于T5，寻找tokenizer_2文件夹
            tokenizer_path = os.path.join(parent_dir, "tokenizer_2")
        
        # 检查tokenizer路径是否存在
        if os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
            print(f"找到本地{model_type.upper()}tokenizer: {tokenizer_path}")
            return tokenizer_path
        else:
            # 如果找不到对应的tokenizer文件夹，回退到模型路径
            print(f"未找到专用{model_type.upper()}tokenizer文件夹，使用模型路径: {model_path}")
            return model_path

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
