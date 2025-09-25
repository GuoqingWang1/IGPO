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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveBatchRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False, default_batchsize=16, val_type='llm'):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, directly return rm score. Otherwise, compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        batch_size = min(default_batchsize, len(data))  # limit batch size in case of smaller data
        
        # Split data into chunks according to batch_size
        batched_inputs = []
        for start_idx in range(0, len(data), batch_size):
            if start_idx + batch_size<len(data):
                batch = data[start_idx:start_idx + batch_size]
            else:
                batch =  data[start_idx:]
            
            prompts, responses, ground_truths, data_sources, extra_infos = [], [], [], [], []
            valid_response_lengths = []

            # Process each item in the batch
            for data_item in batch:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # Decode prompts and responses
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                valid_response_lengths.append(valid_response_length)
                prompts.append(prompt_str)
                responses.append(response_str)
                ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                data_sources.append(data_item.non_tensor_batch[self.reward_fn_key])
                extra_infos.append(data_item.non_tensor_batch.get("extra_info", None))
            
            # Collect inputs for compute_score for the current batch
            inputs = {
                "valid_response_lengths":valid_response_lengths,
                "prompts": prompts,
                "responses": responses,
                "ground_truths": ground_truths,
                "data_sources": data_sources,
                "extra_infos": extra_infos,
                "start_idx": start_idx,
            }
            batched_inputs.append(inputs)
        
        # Process each batch
        for inputs in batched_inputs:
            scores = self.compute_score(
                data_source=inputs["data_sources"],
                prompt_str=inputs["prompts"],
                solution_str=inputs["responses"],
                ground_truth=inputs["ground_truths"],
                extra_info=inputs["extra_infos"],
                val_type=val_type,
                batch_size=batch_size,
            )
            start_idx = inputs['start_idx']
            # Map back scores to the reward tensor
            for idx, score in enumerate(scores):
                response_idx = inputs["valid_response_lengths"][idx] - 1  # last token
                reward_tensor[start_idx + idx, response_idx] = score

                data_source = inputs["data_sources"][idx]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", inputs["prompts"][idx])
                    print("[response]", inputs["responses"][idx])
                    print("[data_source]", data_source, "[ground_truth]", inputs["ground_truths"][idx])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
