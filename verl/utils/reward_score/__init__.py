# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# from .math import math_compute_score
from .r1v import r1v_compute_score
from .gsm8k import gsm8k_compute_score
from .instruction_reward import instruction_compute_score,instruction_val_compute_score


__all__ = [ "r1v_compute_score","gsm8k_compute_score","instruction_compute_score","instruction_val_compute_score"]
