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
# from . import gsm8k, math, prime_math, prime_code
import json

def _default_compute_score(data_source, prompt_str, solution_str, ground_truth, extra_info=None, val_type='f1', info_gain_reward=None, batch_size=1, tokenizer=None, is_validation=False):
    
    
    if type(data_source) != str:
        reslist = []
        for data_source_e,solution_str_e,ground_truth_e in zip(data_source, solution_str, ground_truth):
             res = info_gain.compute_score(solution_str_e, ground_truth_e, data_source, val_type=val_type, info_gain_reward=info_gain_reward, tokenizer=tokenizer, is_validation=is_validation)
        return reslist 
    else:
        res = info_gain.compute_score(solution_str, ground_truth, data_source, val_type=val_type, info_gain_reward=info_gain_reward, tokenizer=tokenizer, is_validation=is_validation)
        return res


