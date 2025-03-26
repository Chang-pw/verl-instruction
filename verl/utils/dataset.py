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
# EXAMPLE = """
# Instruction: 

# In this task, you're given an article and an answer. Your task is to generate the question for the answer based on the given article. 
#  input : Article: Discover Artistic Talents across the Country Redtory, Guangzhou Redtory, established in 2009, is a spacious art district often referred to as Guangzhou's 798. Made up of more than 30 old halls, mostly buildings of red brick, the place was originally built in 1956 as Chinese largest can factory. Some of the rusty machines from the old factory have been kept as art equipment, giving the zone an old charm. Where: No. 128 Yuancun Siheng Road, Tianhe District, Guangzhou Opening hours: Mon-Sun 10:30 am-9 pm Tank Loft, Chongqing This 12,000-square-meter art center was established on the site of a used military tank warehouse . It is the largest contemporary art base in western China, featuring modern exhibition, galleries, design studios and art bookstores. The art zone is located on the campus of the Sichuan Fine Arts Institute and was created by the school in the early 2000s. Where: No. 108 Huangjueping Center Street, Jiulongpo District, Chongqing Opening hours: 9 am-5 pm(closed Mondays) M50, Shanghai M50, also known as 50 Moganshan Road, is the center of Shanghai's contemporary art scene, with more than 100 artists whose studios are open to the public. This district, on the bank of Suzhou Creek, was previously a textile factory until it seized production in 1999. Xue Song, a Shanghai artist, was the first to establish a studio there in 2002. Where: No. 50 Moganshan Road, Pudong District, Shanghai Opening hours: 10 am-5 pm(closed Mondays) 798 Art Zone, Beijing Attracted by ordered designing, convenient traffic and unique style of Bauhaus architecture, art organizations and artists have crowded in the vacant plants and transformed them since 2001, gradually forming a district gathered galleries, art studios, cultural companies, fashion shops etc. As the earliest area where the organizations and artists moved in located in the original area of 798 factory, this place was named as Beijing 798 Art Zone. Where: No. 4 Jiuxianqiao Road, Chaoyang District, Beijing Opening hours: 10 am-6 pm 
#  Answer: Redtory, Guangzhou. Please use JSON format. Don't forget to include the keywords established. 

# Output:

#  <think>

# I'm given an article and an answer, and I need to generate a question that would lead to the answer based on the article. Let me analyze what I have:

# **Article**: The article is about various art districts across China including:
# 1. Redtory, Guangzhou
# 2. Tank Loft, Chongqing
# 3. M50, Shanghai
# 4. 798 Art Zone, Beijing

# **Answer**: "Redtory, Guangzhou." with instructions to use JSON format and include the keyword "established".

# So I need to create a question that:
# 1. Is about Redtory, Guangzhou
# 2. Involves the word "established"
# 3. Has an answer specifically of "Redtory, Guangzhou"
# 4. Should be in JSON format

# Looking at the article section about Redtory, the key information is:
# - Established in 2009
# - Spacious art district referred to as Guangzhou's 798
# - Made up of more than 30 old halls
# - Originally built in 1956 as Chinese largest can factory
# - Location and opening hours

# A question that would lead to the answer "Redtory, Guangzhou" and uses the word "established" could be something like:

# "Which art district was established in 2009 and is often referred to as Guangzhou's 798?"

# Since I need to provide this in JSON format, I'll structure it like:

# ```json
# {
#   "question": "Which art district was established in 2009 and is often referred to as Guangzhou's 798?"
# }
# ```

# This question specifically targets Redtory, Guangzhou, uses the keyword "established", and would naturally have "Redtory, Guangzhou" as its answer.
# </think>
# <answer>
# Based on the article and the given answer "Redtory, Guangzhou," I need to create a question that would elicit this specific response while including the keyword "established" and formatting in JSON.

# ```json
# {
#   "question": "Which art district was established in 2009 and is often referred to as Guangzhou's 798?"
# }
# ```

# This question directly targets the information in the article about Redtory being established in 2009 and being referred to as "Guangzhou's 798," making "Redtory, Guangzhou" the clear and specific answer.
# </answer>
#     """.strip()

# intruction_system_prompt = """You are a meticulous assistant who precisely adheres to all explicit and implicit constraints in user instructions. When presented with complex, multi-constraint tasks, you follow a structured approach:

# FORMAT: Your response must be structured with two distinct sections:
# <think> Your detailed analytical process and strategy development </think>
# <answer> Your final solution that adheres to all requirements </answer>

# In your <think> section, employ these strategic approaches:
# 1. Constraint Analysis: Thoroughly examine each explicit and implicit constraint, identifying their interdependencies and potential conflicts.
# 2. Validation Strategy: Draft preliminary answers, then methodically verify compliance with each constraint, making iterative refinements as needed.
# 3. Sequencing Logic: Determine the optimal order for addressing constraints, prioritizing foundational requirements before tackling dependent ones.
# 4. Edge Case Consideration: Proactively identify boundary conditions and exceptions that might challenge constraint compliance.
# 5. Coherence Check: Ensure your solution maintains internal consistency while satisfying all requirements simultaneously.
# 6. Format Verification: Confirm your response adheres to all specified formatting and structural guidelines.

# In your <answer> section, deliver a solution that precisely implements all requirements while maintaining natural flow and coherence. Your final response must satisfy all constraints without drawing attention to the mechanics of constraint management itself.

# Here is an example: 

# {example}
# """.format(example = EXAMPLE).strip()

intruction_system_prompt = '''You are a helpful assistant who rigorously follows all explicit and implicit constraints in user instructions. When the user provides an instruction containing multiple constraints, you first analyze how to best follow it. During the thinking process, you conduct a comprehensive constraint analysis and then develop execution strategies that fully comply with all requirements. Finally, you provide the user with an answer that adheres to all specified formats and rules. Both the thinking process and answer must be enclosed within the respective tags:
<think>
The thinking process for constraint analysis and execution strategies
</think>

<answer>
An answer adhering to all specified formats and rules
</answer>
'''

# print(system_prompt)
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: Union[Dict[str, Any], ImageObject], max_pixels: int, min_pixels: int) -> ImageObject:
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        # answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = "",
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        # self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = intruction_system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        # messages = [{"role": "user", "content": row_dict[self.prompt_key] +  '\n\n' + 'Remember: We will check for the "<think>.*</think><answer>.*</answer>" pattern to ensure compliance.'}]
        messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(image, self.max_pixels, self.min_pixels) for image in row_dict.pop(self.image_key)
                ]
            }
            model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        instruction_id_list = row_dict.pop('instruction_id_list')
        kwargs = row_dict.pop('kwargs')
        row_dict["ground_truth"] = {
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs,
        }
        return row_dict
