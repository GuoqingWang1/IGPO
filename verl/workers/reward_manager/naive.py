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
import json
from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False, val_type='f1', info_gain_rewards=None, is_validation=False):
        """We will expand this function gradually based on the available datasets"""
        data_str = str(data)
        if is_validation:
            f1_scores = []
            em_scores = []
            noformatf1_scores = []
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # info_gain_reward   
            info_gain_reward = info_gain_rewards[i]

            score = self.compute_score(
                data_source=data_source,
                prompt_str = prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                val_type=val_type,
                info_gain_reward=info_gain_reward,
                tokenizer=self.tokenizer,
				is_validation=is_validation,
            )

            if is_validation:
                f1_scores.append(score['f1'])
                em_scores.append(score['em'])
                noformatf1_scores.append(score['noformatf1'])
                reward_tensor[i, :valid_response_length] = torch.tensor(score['scores'])
            else:
                reward_tensor[i, :valid_response_length] = torch.tensor(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and val_type == 'f1':
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[data_source]",data_source,"[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if is_validation:
            return {
                "f1_scores": f1_scores,
                "em_scores": em_scores,
				"noformatf1_scores": noformatf1_scores,
				"reward_tensor": reward_tensor,
            }
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }	
        else:
            return reward_tensor
