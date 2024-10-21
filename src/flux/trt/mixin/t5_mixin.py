from typing import Any
from flux.trt.mixin.base_mixin import BaseMixin


class T5Mixin(BaseMixin):
    def __init__(
        self,
        text_maxlen: int,
        hidden_size: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_maxlen = text_maxlen
        self.hidden_size = hidden_size


    def get_mixin_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "text_maxlen": self.text_maxlen,
            "hidden_size": self.hidden_size,
        }
