from abc import ABC, abstractmethod
from typing import Any


class BaseMixin(ABC):
    @abstractmethod
    def get_mixin_params(self) -> dict[str, Any]:
        pass
