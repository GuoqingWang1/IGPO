# =============================================================================
# Based on the Search-R1 example from the Search-R1 project.
#
# Original Authors: Jin Bowen, Zeng Hansi, Yue Zhenrui, Wang Dong, Zamani Hamed, Han Jiawei
#
# License: Apache 2.0
# Project URL: https://github.com/PeterGriffinJin/Search-R1
# =============================================================================

import torch
import copy
import re
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tensordict import TensorDict
from scrl.llm_agent.tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
import numpy as np
import traceback
import torch.nn.functional as F
from tools_server.util import MessageClient
from tools_server.initialize_prompts import SYSTEM_PROMPT


@dataclass
class GenerationConfig:
    max_turns: int
    num_gpus: int
    data_writing_path: str = None
    model_name: str = None
    n: int = 1
    project_name: str = None
    experiment_name: str = None
    search_engine: str = "rag"
    nnodes: int = 1
    oss_access_key_id: str = ''
    oss_access_key_secret: str = ''
    oss_endpoint: str = ''
    system_prompt: Optional[str] = SYSTEM_PROMPT
    codeact_env_disabled: bool = True
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        client = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.system_prompt = config.system_prompt
        self.codeact_env_disabled = config.codeact_env_disabled

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id
        ))

        self.client = client

    def _update_right_side(self, original_right_side: Dict, 
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side of rollings."""
        if next_obs_ids is not None:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses, next_obs_ids],
                pad_to_left=False
            )
        else:
            responses = self.tensor_fn.concatenate_with_padding(
                [original_right_side['responses'], cur_responses],
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        
        return {'responses': responses[:, :effective_len]}

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        return DataProto.from_dict({
                'input_ids': new_input_ids[:, -effective_len:],
                'position_ids': new_position_ids[:, -effective_len:],
                'attention_mask': new_attention_mask[:, -effective_len:]
            })

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        return next_obs_ids
        
    def postprocess_predictions(self, rollings_active: DataProto, gen_output: DataProto) -> Tuple[List[int], List[bool]]:
        """Postprocess predictions to remove padding and convert to list of strings."""
        """return: list of query contents including history"""

        pass
        return [{"prompt":""} for _ in range(rollings_active.batch['input_ids'].shape[0])]


    def execute_predictions(
        self, tool_call_list, total_number
    ) :
        query_contents = [{"idx": tool_call[0], "question": tool_call[1], "think": tool_call[2],
                           "tool_call": tool_call[3], "total_number":total_number} for tool_call in tool_call_list]

        query_contents = self.client.submit_tasks(query_contents)
        return query_contents

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus * self.config.nnodes
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            output = self.actor_rollout_wg.generate_sequences(active_batch)
            return output
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output



    def parse_question(self, input_ids: torch.Tensor) -> str:
        """Parse question to get the query content."""
        query_contents = self.tokenizer.batch_decode(input_ids)
        query_contents = [re.sub(r'^(<\|endoftext\|>)+', '', content) for content in query_contents]
        query_contents = [content.split("<|im_start|>user\n")[1].split("<|im_end|>")[0] for content in query_contents]
        return query_contents

    def parse_response(self, input_ids: torch.Tensor, think: bool = False) -> List[Tuple[bool, str, str]]:
        """Parse response to get the thinking process and answer or tool call.
            return: [(is_stop, thinking, answer/tool_call), ...]
        """
        response_contents = self.tokenizer.batch_decode(input_ids)
        results = []
        for i, content in enumerate(response_contents):
            if think:
                content = "<think>" + content
            if "<think>" in content and "<answer>" in content:
                if "</think>" not in content or "</answer>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    answer = content.split("<answer>")[1].split("</answer>")[0]
                    results.append((True, think, answer))
            elif "<think>" in content and "<tool_call>" in content and self.codeact_env_disabled:
                if "</tool_call>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                    try:
                        tool_call = json.loads(tool_call)
                        assert "name" in tool_call, "no vliad function name in tool_call"
                        assert "arguments" in tool_call, "no valid arguments in tool_call"
                        assert tool_call["name"] not in [""], "invalid tool name"
                        results.append((False, think, tool_call))
                    except Exception as e:
                        if i < 10:
                            print(f"model tool call format error: {e}")
                            print(content.replace('<|endoftext|>',''))
                        results.append((True, "", ""))
            elif "<think>" in content and "<code>" in content and not self.codeact_env_disabled:  
                if "</code>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think = content.split("<think>")[1].split("</think>")[0]
                    code = content.split("<code>")[1].split("</code>")[0]
                    try:
                        tool_call = {"name":"code_act", "arguments":{"code":code}}
                        results.append((False, think, tool_call))
                    except Exception as e:
                        if i < 10:
                            print(f"model tool call format error: {e}")
                            print(content.replace('<|endoftext|>',''))
                        results.append((True, "", ""))
            else:
                results.append((True, "", ""))
        return results

    def pseudo_generate_sequences(self, prompts, response):
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        eos_token_id = self.tokenizer.eos_token_id
        non_tensor_batch = prompts.non_tensor_batch
        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=2000).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def run_llm_loop(self, gen_batch: DataProto, global_steps: int, ground_truths: list) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""


        node_rank = int(os.environ["PET_NODE_RANK"])
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} * {self.config.n} datas!",flush=True)
        query_contents = self.parse_question(gen_batch.batch['input_ids'])
        messages_list = []
        agent_grpo_idx = []
        for gt in ground_truths:
            if "<|answer_split|>" in gt['ground_truth']:
                gt['ground_truth'] = gt['ground_truth'].split("<|answer_split|>")[0]
            _gt = gt['ground_truth'].strip()
            if _gt.startswith('['):
                parsed_gt = json.loads(_gt)
                if isinstance(parsed_gt, list):
                    label = 'true'
                    for item in parsed_gt:
                        if item['label'].lower() == 'false':
                            label = 'false'
                            break
                    gt['ground_truth'] = label

        # Preprocess ground_truths to align with batch_size * n
        ground_truths_rolling = []
        for gt in ground_truths:
            for _ in range(self.config.n):
                ground_truths_rolling.append(gt)

        pseudo_resps_with_gt = [self.tokenizer(f"\nNow there's enough information to answer\n</think>\n<answer>\n{ground_truth['ground_truth']}\n</answer><|im_end|>", return_tensors="pt")['input_ids'].tolist()[0] for ground_truth in ground_truths_rolling]
        for i, resp in enumerate(pseudo_resps_with_gt):
            # print(f"resp[{i}] length: {len(resp)}, content: {resp[:10]}...")  
            if len(resp) == 0:
                print(f"❗❗❗ EMPTY at index {i}, ground_truth: '{ground_truths_rolling[i]['ground_truth']}'")
        len_st = len(self.tokenizer("\nNow there's enough information to answer\n</think>\n<answer>\n", return_tensors="pt")['input_ids'].tolist()[0])
        len_ed = len(self.tokenizer("\n</answer><|im_end|>", return_tensors="pt")['input_ids'].tolist()[0])
        gt_idx = []
        for resp_with_gt in pseudo_resps_with_gt:
            idx = [len_st, len(resp_with_gt) - len_ed]
            gt_idx.append(idx)
        

        for idx, query_content in enumerate(query_contents):
            for _ in range(self.config.n):
                if self.system_prompt:
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": query_content}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": query_content}
                    ]
                messages_list.append(messages)
                agent_grpo_idx.append(idx)
        activate_list = [i for i in range(len(messages_list))]
        message_string_list = ["" for _ in range(len(messages_list))]

        # Ensure the directory exists
        output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
        if not os.path.exists(output_dir):
            print(f"Directory not exist, create at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Create an information gain reward: List[List]
        gt_log_probs_per_turn = [[] for _ in range(len(messages_list))]
        gt_entropys_per_turn = [[] for _ in range(len(messages_list))]
        info_gain_rewards = [[] for _ in range(len(messages_list))]


        for step in range(self.config.max_turns):
            print(f"node {node_rank} step {step} start!")
            activate_messages_list = [messages_list[i] for i in activate_list]

            if activate_list == []:
                break
            try:
                rollings_active = self.tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tokenize=False)
            except:
                json.dump(activate_messages_list,open('./debug.json','w'))
                for e in activate_messages_list:
                    print(e)
                    rollings_active = self.tokenizer.apply_chat_template([e], add_generation_prompt=True, tokenize=False)
    
            think = True
            
            if think:
                rollings_active = [rolling + "<think>" for rolling in rollings_active]
            else:
                rollings_active = [rolling for rolling in rollings_active]
            
            rollings_active = self.tokenizer(rollings_active, return_tensors="pt", padding=True)
                
            pad_mask = rollings_active['input_ids'] != self.tokenizer.pad_token_id
            sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
            rollings_active['input_ids'] = rollings_active['input_ids'].gather(1, sorted_indices)
            rollings_active['attention_mask'] = rollings_active['attention_mask'].gather(1, sorted_indices)
            
            attention_mask = rollings_active['attention_mask']
            rollings_active['position_ids'] = self.tensor_fn.create_position_ids(attention_mask)

            print(f"node {node_rank}, turn {step} rollings_active is {len(rollings_active['input_ids'])} datas")
            rollings_active = DataProto.from_dict({
                'input_ids': rollings_active['input_ids'],
                'attention_mask': rollings_active['attention_mask'],
                'position_ids': rollings_active['position_ids'],
            })
            
            if step == 0:
                info_gain_rollings_active = copy.deepcopy(rollings_active)
            else:
                if info_gain_rollings_active.batch['input_ids'].shape[1] < rollings_active.batch['input_ids'].shape[1]:
                    info_gain_rollings_active.batch['input_ids'] = F.pad(
                        info_gain_rollings_active.batch['input_ids'], 
                        pad=(0, rollings_active.batch['input_ids'].shape[1] - info_gain_rollings_active.batch['input_ids'].shape[1]),
                        mode='constant',
                        value=self.tokenizer.pad_token_id,
                       )
					
                    info_gain_rollings_active.batch['attention_mask'] = F.pad(
                        info_gain_rollings_active.batch['attention_mask'], 
                        pad=(0, rollings_active.batch['attention_mask'].shape[1] - info_gain_rollings_active.batch['attention_mask'].shape[1]),
                        mode='constant',
                        value=self.tokenizer.pad_token_id,
                        )

                    info_gain_rollings_active.batch['position_ids'] = F.pad(
                        info_gain_rollings_active.batch['position_ids'], 
                        pad=(0, rollings_active.batch['position_ids'].shape[1] - info_gain_rollings_active.batch['position_ids'].shape[1]),
                        mode='constant',
                        value=self.tokenizer.pad_token_id,
                        )
                
                    for i in range(len(activate_list)):
                        info_gain_rollings_active.batch['input_ids'][activate_list[i], :] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][activate_list[i], :] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][activate_list[i], :] = rollings_active.batch['position_ids'][i]
                else:
                    for i in range(len(activate_list)):   
                        info_gain_rollings_active.batch['input_ids'][activate_list[i], :len(rollings_active.batch['input_ids'][i])] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][activate_list[i], :len(rollings_active.batch['attention_mask'][i])] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][activate_list[i], :len(rollings_active.batch['position_ids'][i])] = rollings_active.batch['position_ids'][i]    
            
            
            pseudo_gen_output = self.pseudo_generate_sequences(info_gain_rollings_active, pseudo_resps_with_gt)            
            pseudo_gen_output_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)
            
            if step == 0:
                gt_probs = {}
                for i in activate_list:
                    gt_probs[i] = torch.exp(pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]].mean()).item()
                    gt_log_probs_per_turn[i].append(pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())
                    gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())
            else:
                for i in activate_list: 
                    cur_prob = torch.exp(pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]].mean()).item()      
                    info_gain_rewards[i].append(cur_prob - gt_probs[i])
                    gt_probs[i] = cur_prob
                    gt_log_probs_per_turn[i].append(pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())
                    gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())       

            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            meta_info = gen_output.meta_info
            print(f"node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas")

            results = self.parse_response(gen_output.batch['responses'], think=think)
            assert len(results) == len(activate_list) # After each round of updates, the number of results matches the number of currently active queries
            activate_list_copy = []
            tool_call_list = []
            for i in range(len(results)):
                if results[i][0]:
                    message_string_list[activate_list[i]] = self.tokenizer.decode(rollings_active.batch['input_ids'][i], skip_special_tokens=False).replace("<|endoftext|>", "") + self.tokenizer.decode(gen_output.batch['responses'][i], skip_special_tokens=False).replace("<|endoftext|>", "")
                else:
                    activate_list_copy.append(activate_list[i])
                    tool_call_list.append((activate_list[i], messages_list[activate_list[i]][1]["content"], results[i][1], results[i][2]))
                    
            tool_call_list = self.execute_predictions(tool_call_list,len(messages_list))
            print(f"node {node_rank}, turn {step} tool_call_list {len(tool_call_list)} datas")
            for i in range(len(tool_call_list)):
                if not self.codeact_env_disabled:  
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant", 
                            "content": "<think>" + tool_call_list[i]['think'] + "</think>"+"\n<code>" + str(tool_call_list[i]['tool_call']['arguments']['code']) + "</code>", 
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user", 
                                "content": "<code_response>" + tool_call_list[i]['content'] + "</code_response>",
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "user", 
                                "content": "<code_response>" + 'tool format error' + "</code_response>",
                            }
                        )
                else:
                    messages_list[tool_call_list[i]['idx']].append(
                        {
                            "role": "assistant", 
                            "content": "<think>" + tool_call_list[i]['think'] + "</think>", 
                            "tool_calls": [
                                            {
                                                "type": "function", 
                                                "function": tool_call_list[i]['tool_call']
                                            }
                                        ]
                        }
                    )
                    try:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool", 
                                "name": tool_call_list[i]['tool_call']['name'],
                                "content": tool_call_list[i]['content']
                            }
                        )
                    except:
                        messages_list[tool_call_list[i]['idx']].append(
                            {
                                "role": "tool", 
                                "name": '',
                                "content": 'tool format error'
                            }
                        )
            print(f"Round {step} finished. Node {node_rank} originally had {len(activate_list)} queries, now has {len(activate_list_copy)} queries.")
            activate_list = activate_list_copy

        if activate_list != []:
            for i in activate_list:
                message_string_list[i] = self.tokenizer.apply_chat_template(messages_list[i], add_generation_prompt=True, tokenize=False)
        
        response_str_list = []
        initial_prompt_list = []
        for i, messages in enumerate(messages_list):
            initial_prompt = self.tokenizer.apply_chat_template(messages[0:2], add_generation_prompt=True, tokenize=False)
            initial_prompt_list.append(initial_prompt)
            response_str_list.append(message_string_list[i][len(initial_prompt):])
        
        prompts_tokenizered = self.tokenizer(initial_prompt_list, return_tensors="pt",padding=True)

        prompts_repeated = prompts_tokenizered['input_ids']
        pad_mask = prompts_repeated != self.tokenizer.pad_token_id
        sorted_indices = pad_mask.to(torch.int64).argsort(dim=1, stable=True)
        

        prompts_repeated = prompts_repeated.gather(1, sorted_indices)
        prompts_attention_mask = prompts_tokenizered['attention_mask'].gather(1, sorted_indices)

        responses = self.tokenizer(response_str_list, return_tensors="pt",padding=True)['input_ids']
        
        responses_attention_mask = self.tokenizer(response_str_list, return_tensors="pt",padding=True)['attention_mask']
        attention_mask = torch.cat((prompts_attention_mask, responses_attention_mask), dim=-1)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        
        message_tensor = DataProto.from_dict({
            'prompts': prompts_repeated,
            'responses': responses,
            'input_ids': torch.cat((prompts_repeated, responses), dim=-1),
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        message_tensor.meta_info.update(meta_info)
        message_tensor.non_tensor_batch['agent_grpo_idx'] = np.array(agent_grpo_idx, dtype=object)
        print("generation finished")


        
        print(f"node {node_rank} message_string_list {len(message_string_list)}")
        
        return message_string_list, message_tensor, info_gain_rewards
    
