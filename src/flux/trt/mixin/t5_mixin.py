#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from torch import dtype
from flux.trt.mixin.base_mixin import BaseMixin


class T5Mixin(BaseMixin):
    def __init__(
        self,
        text_maxlen: int,
        hidden_size: int,
        should_be_dtype: dtype,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_maxlen = text_maxlen
        self.hidden_size = hidden_size
        self.should_be_dtype = should_be_dtype

    def get_mixin_params(self) -> dict[str, Any]:
        """helper class that return the parameters used for construction"""
        return {
            "text_maxlen": self.text_maxlen,
            "hidden_size": self.hidden_size,
            "should_be_dtype": self.should_be_dtype,
        }
