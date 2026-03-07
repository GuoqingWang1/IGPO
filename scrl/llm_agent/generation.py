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

# Verification flag for vectorized GT LogProb computation
# Set IGPO_VERIFY_VECTORIZED=true to enable comparison between vectorized and original mode
VERIFY_VECTORIZED = os.environ.get('IGPO_VERIFY_VECTORIZED', '').lower() in ('true', '1', 'yes')
if VERIFY_VECTORIZED:
    print("[IGPO] Verification mode enabled: will compare vectorized vs original results")

# Import vectorized GT logprob module
from scrl.llm_agent.vectorized_gt_logprob import (
    is_vectorized_enabled,
    VectorizedGTConfig,
    VectorizedGTLogProbComputer,
)
import math

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
    search_engine: str = "online_search"
    nnodes: int = 1
    oss_access_key_id: str = ''
    oss_access_key_secret: str = ''
    oss_endpoint: str = ''
    system_prompt: Optional[str] = SYSTEM_PROMPT
    codeact_env_disabled: bool = True
    # info_gain_type: "prob_diff" (probability difference) or "log_prob_diff" (log probability difference)
    info_gain_type: str = "prob_diff"
    

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
                    think_content = content.split("<think>")[1].split("</think>")[0]
                    answer = content.split("<answer>")[1].split("</answer>")[0]
                    results.append((True, think_content, answer))
            elif "<think>" in content and "<tool_call>" in content and self.codeact_env_disabled:
                if "</tool_call>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think_content = content.split("<think>")[1].split("</think>")[0]
                    tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                    try:
                        tool_call = json.loads(tool_call)
                        assert "name" in tool_call, "no vliad function name in tool_call"
                        assert "arguments" in tool_call, "no valid arguments in tool_call"
                        assert tool_call["name"] not in [""], "invalid tool name"
                        results.append((False, think_content, tool_call))
                    except Exception as e:
                        if i < 10:
                            print(f"model tool call format error: {e}")
                            print(content.replace('<|endoftext|>',''))
                        results.append((True, "", ""))
            elif "<think>" in content and "<code>" in content and not self.codeact_env_disabled:  # code act format
                if "</code>" not in content or "</think>" not in content:
                    results.append((True, "", ""))
                else:
                    think_content = content.split("<think>")[1].split("</think>")[0]
                    code = content.split("<code>")[1].split("</code>")[0]
                    try:
                        tool_call = {"name":"code_act", "arguments":{"code":code}}
                        results.append((False, think_content, tool_call))
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
        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        last_valid_pos_ids = (attention_mask.sum(dim=1, keepdim=True).long() - 1).clamp(min=0)
        response_position_ids = last_valid_pos_ids + delta_position_id
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
        print(f"node {node_rank} gains {len(gen_batch.batch['input_ids'])} * {self.config.n} datas!", flush=True)
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

        # Use offset_mapping to precisely calculate ground truth token range
        # Avoid index offset caused by subword tokenization boundary effects
        PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX = "\n</answer><|im_end|>"
        
        pseudo_resps_with_gt = []
        gt_idx = []
        
        for ground_truth in ground_truths_rolling:
            gt_text = ground_truth['ground_truth']
            full_text = f"{PREFIX}{gt_text}{SUFFIX}"
            
            # Use offset_mapping to get precise character-token mapping
            encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
            token_ids = encoding['input_ids'].tolist()[0]
            offset_mapping = encoding['offset_mapping'].tolist()[0]  # [(char_start, char_end), ...]
            
            pseudo_resps_with_gt.append(token_ids)
            
            if len(token_ids) == 0:
                print(f"❗❗❗ EMPTY token_ids for ground_truth: '{gt_text}'")
                gt_idx.append([0, 0])
                continue
            
            # Calculate ground truth position in original string
            gt_char_start = len(PREFIX)
            gt_char_end = len(PREFIX) + len(gt_text)
            
            # Find precise token indices through offset_mapping
            gt_token_start = None
            gt_token_end = None
            
            for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                # Find the first token covering gt_char_start
                if gt_token_start is None and char_end > gt_char_start:
                    gt_token_start = token_idx
                # Find the last token covering gt content (char_start < gt_char_end)
                if char_start < gt_char_end and char_end > 0:
                    gt_token_end = token_idx + 1
            
            # Boundary check
            if gt_token_start is None:
                gt_token_start = len(token_ids)
            if gt_token_end is None:
                gt_token_end = len(token_ids)
            
            gt_idx.append([gt_token_start, gt_token_end])
        

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

        # Ensure output directory exists (only create when project_name and experiment_name are valid)
        output_dir = None
        if self.config.project_name and self.config.experiment_name:
            output_dir = f"./outputs/{self.config.project_name}/{self.config.experiment_name}/rollout"
            if not os.path.exists(output_dir):
                print(f"Directory not exist, create at {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
        
        # Create information gain reward: list[list]
        gt_log_probs_per_turn = [[] for _ in range(len(messages_list))]
        gt_entropys_per_turn = [[] for _ in range(len(messages_list))]
        info_gain_rewards = [[] for _ in range(len(messages_list))]
        gt_values = {}  # Store previous turn's value (probability or log probability, depending on info_gain_type)

        # Vectorized switch detection
        use_vectorized_gt_logprob = is_vectorized_enabled()
        
        # ========== Vectorized computation: data collection structure ==========
        # When vectorization is enabled, delay GT log probs computation for batch processing after loop
        vectorized_data_collector = None
        # For verification: also compute original mode results when VERIFY_VECTORIZED is enabled
        original_mode_gt_values = {} if (use_vectorized_gt_logprob and VERIFY_VECTORIZED) else None
        original_mode_info_gains = [[] for _ in range(len(messages_list))] if (use_vectorized_gt_logprob and VERIFY_VECTORIZED) else None
        
        if use_vectorized_gt_logprob:
            vectorized_data_collector = {
                'pseudo_outputs_per_turn': [],  # List of pseudo_gen_output for each turn
                'activate_lists_per_turn': [],  # activate_list for each turn
                'gt_idx': gt_idx,  # GT token range
                'num_samples': len(messages_list),
            }
            print(f"[IGPO] Vectorized GT LogProb: Collecting data for batch computation...")
            if VERIFY_VECTORIZED:
                print(f"[IGPO] Verification mode: will also compute original mode for comparison")

        for step in range(self.config.max_turns):
            print(f"node {node_rank} step {step} start!")
            activate_messages_list = [messages_list[i] for i in activate_list]

            if activate_list == []:
                break
            try:
                rollings_active = self.tokenizer.apply_chat_template(activate_messages_list, add_generation_prompt=True, tokenize=False)
            except Exception as e:
                print(f"Error in tokenizer.apply_chat_template: {e}")
                # Fallback strategy: process each message individually
                rollings_active = []
                for msg in activate_messages_list:
                    try:
                        result = self.tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
                        rollings_active.extend(result)
                    except Exception as inner_e:
                        print(f"Failed to process message: {inner_e}")
                        raise  # Cannot recover, raise exception
    
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
                        value=0,  # attention_mask uses 0 for padding positions
                        )

                    info_gain_rollings_active.batch['position_ids'] = F.pad(
                        info_gain_rollings_active.batch['position_ids'], 
                        pad=(0, rollings_active.batch['position_ids'].shape[1] - info_gain_rollings_active.batch['position_ids'].shape[1]),
                        mode='constant',
                        value=0,  # position_ids uses 0 for padding
                        )
                
                    for i in range(len(activate_list)):
                        info_gain_rollings_active.batch['input_ids'][activate_list[i], :] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][activate_list[i], :] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][activate_list[i], :] = rollings_active.batch['position_ids'][i]
                else:
                    src_len = rollings_active.batch['input_ids'].shape[1]
                    ig_len = info_gain_rollings_active.batch['input_ids'].shape[1]
                    for i in range(len(activate_list)):
                        idx = activate_list[i]
                        info_gain_rollings_active.batch['input_ids'][idx, :src_len] = rollings_active.batch['input_ids'][i]
                        info_gain_rollings_active.batch['attention_mask'][idx, :src_len] = rollings_active.batch['attention_mask'][i]
                        info_gain_rollings_active.batch['position_ids'][idx, :src_len] = rollings_active.batch['position_ids'][i]
                        if src_len < ig_len:
                            info_gain_rollings_active.batch['input_ids'][idx, src_len:] = self.tokenizer.pad_token_id
                            info_gain_rollings_active.batch['attention_mask'][idx, src_len:] = 0
                            info_gain_rollings_active.batch['position_ids'][idx, src_len:] = 0
            
            pseudo_gen_output = self.pseudo_generate_sequences(info_gain_rollings_active, pseudo_resps_with_gt)
            
            # ========== GT LogProb computation (vectorized or immediate) ==========
            if use_vectorized_gt_logprob and vectorized_data_collector is not None:
                # Vectorized mode: collect data, delay computation
                # Save current turn's pseudo_gen_output (need clone to avoid modification by subsequent operations)
                pseudo_output_clone = DataProto.from_dict({
                    'prompts': pseudo_gen_output.batch['prompts'].clone(),
                    'responses': pseudo_gen_output.batch['responses'].clone(),
                    'input_ids': pseudo_gen_output.batch['input_ids'].clone(),
                    'attention_mask': pseudo_gen_output.batch['attention_mask'].clone(),
                    'position_ids': pseudo_gen_output.batch['position_ids'].clone(),
                })
                vectorized_data_collector['pseudo_outputs_per_turn'].append(pseudo_output_clone)
                vectorized_data_collector['activate_lists_per_turn'].append(list(activate_list))
                
                # ========== Verification: also compute original mode for comparison ==========
                if VERIFY_VECTORIZED and original_mode_gt_values is not None:
                    # Compute log_probs using original mode (immediate computation)
                    verify_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)
                    info_gain_type = self.config.info_gain_type
                    
                    if step == 0:
                        for i in activate_list:
                            if gt_idx[i][0] >= gt_idx[i][1]:
                                continue
                            log_probs = verify_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                            mean_log_prob = log_probs.mean().item()
                            if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                                continue
                            if info_gain_type == "log_prob_diff":
                                original_mode_gt_values[i] = mean_log_prob
                            else:
                                original_mode_gt_values[i] = torch.exp(torch.tensor(mean_log_prob)).item()
                    else:
                        for i in activate_list:
                            if gt_idx[i][0] >= gt_idx[i][1]:
                                continue
                            if i not in original_mode_gt_values:
                                continue
                            log_probs = verify_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                            mean_log_prob = log_probs.mean().item()
                            if math.isnan(mean_log_prob):
                                continue
                            if info_gain_type == "log_prob_diff":
                                cur_value = mean_log_prob
                                info_gain = cur_value - original_mode_gt_values[i]
                            else:
                                cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                                info_gain = cur_value - original_mode_gt_values[i]
                            if math.isnan(info_gain) or math.isinf(info_gain):
                                continue
                            original_mode_info_gains[i].append(info_gain)
                            original_mode_gt_values[i] = cur_value
            else:
                # Original mode: immediate computation
                pseudo_gen_output_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_gen_output)
                
                # ========== Compute info_gain_reward based on info_gain_type ==========
                # "prob_diff": Use probability difference exp(mean(log P_t)) - exp(mean(log P_{t-1}))
                # "log_prob_diff": Use log probability difference mean(log P_t) - mean(log P_{t-1})
                
                info_gain_type = self.config.info_gain_type  # "prob_diff" or "log_prob_diff"
                
                if step == 0:
                    for i in activate_list:
                        # Check if gt_idx range is valid
                        if i >= len(gt_idx) or gt_idx[i][0] >= gt_idx[i][1]:
                            continue
                        log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                        mean_log_prob = log_probs.mean().item()
                        
                        # Skip if mean_log_prob is nan or inf
                        if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                            continue
                        
                        if info_gain_type == "log_prob_diff":
                            gt_values[i] = mean_log_prob
                        else:  # "prob_diff" (default)
                            gt_values[i] = torch.exp(torch.tensor(mean_log_prob)).item()
                        
                        gt_log_probs_per_turn[i].append(log_probs.tolist())
                        gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())
                else:
                    for i in activate_list:
                        # Check if gt_idx range is valid
                        if gt_idx[i][0] >= gt_idx[i][1]:
                            # Empty range, skip (don't compute info_gain)
                            continue
                        # Check if gt_values was initialized in step 0, skip if not to avoid KeyError/nan
                        if i not in gt_values:
                            continue
                        log_probs = pseudo_gen_output_log_probs.batch['old_log_probs'][i, gt_idx[i][0]:gt_idx[i][1]]
                        mean_log_prob = log_probs.mean().item()
                        
                        # Check for nan in mean_log_prob
                        if math.isnan(mean_log_prob):
                            info_gain_rewards[i].append(0.0)
                            continue
                        
                        prev_value = gt_values[i]  # Save previous value for verification
                        
                        if info_gain_type == "log_prob_diff":
                            # Use log probability difference
                            cur_value = mean_log_prob
                            info_gain = cur_value - gt_values[i]
                        else:  # "prob_diff" (default)
                            # Use probability difference
                            cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                            info_gain = cur_value - gt_values[i]
                        
                        # Check for nan/inf in info_gain
                        if math.isnan(info_gain) or math.isinf(info_gain):
                            info_gain_rewards[i].append(0.0)
                            continue
                        
                        info_gain_rewards[i].append(info_gain)
                        gt_values[i] = cur_value
                        
                        gt_log_probs_per_turn[i].append(log_probs.tolist())
                        gt_entropys_per_turn[i].append(pseudo_gen_output_log_probs.batch['entropys'][i, gt_idx[i][0]:gt_idx[i][1]].tolist())       

            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            meta_info = gen_output.meta_info
            print(f"node {node_rank}, turn {step} gen_output {len(gen_output.batch['responses'])} datas")

            results = self.parse_response(gen_output.batch['responses'], think=think)
            assert len(results) == len(activate_list)  # After each round update, result count equals active query count
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
                if not self.codeact_env_disabled:  # code act enabled
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
                                "content": "<code_response>" + 'Return format error, code execution failed' + "</code_response>",
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
                                "content": 'Return format error, tool call failed'
                            }
                        )
            print(f"Turn {step} ended, node {node_rank} originally had {len(activate_list)} queries, now has {len(activate_list_copy)} queries")
            activate_list = activate_list_copy
           
        
        # ========== Vectorized GT LogProb batch computation (Prealigned Version) ==========
        # This section uses the prealigned vectorization strategy for better mathematical rigor.
        # It is completely independent and does not affect original mode computation.
        if use_vectorized_gt_logprob and vectorized_data_collector is not None:
            num_turns_collected = len(vectorized_data_collector['pseudo_outputs_per_turn'])
            _USE_LEGACY_VECTORIZED = False  # Legacy implementation is disabled
            if num_turns_collected > 0:
                # Use prealigned vectorized module (lazy import)
                from scrl.llm_agent.prealigned_vectorized import compute_vectorized_gt_logprob
                
                info_gain_type = self.config.info_gain_type
                gt_idx = vectorized_data_collector['gt_idx']
                
                # Call prealigned vectorized computation
                vectorized_result = compute_vectorized_gt_logprob(
                    pseudo_outputs_per_turn=vectorized_data_collector['pseudo_outputs_per_turn'],
                    activate_lists_per_turn=vectorized_data_collector['activate_lists_per_turn'],
                    gt_idx=gt_idx,
                    actor_rollout_wg=self.actor_rollout_wg,
                    tokenizer=self.tokenizer,
                    info_gain_type=info_gain_type,
                    enable_strict_validation=VERIFY_VECTORIZED,
                )
                
                # Update results from vectorized computation
                gt_values = vectorized_result['gt_values']
                info_gain_rewards = vectorized_result['info_gain_rewards']
                gt_log_probs_per_turn = vectorized_result['gt_log_probs_per_turn']
                gt_entropys_per_turn = vectorized_result['gt_entropys_per_turn']
                
                # Store vectorized mean_log_probs for verification (if enabled)
                if VERIFY_VECTORIZED:
                    vectorized_mode_log_probs_per_turn = vectorized_result.get('vectorized_mean_log_probs', None)
                
            if num_turns_collected > 0 and _USE_LEGACY_VECTORIZED:
                # ===== LEGACY: Left-padding implementation (disabled) =====
                print(f"[IGPO] Vectorized GT LogProb: Processing {num_turns_collected} turns in batch...")
                
                # Batch compute GT log probs for all turns
                info_gain_type = self.config.info_gain_type
                all_log_probs_results = []
                
                # Strategy: Reconstruct input_ids so ALL samples have response of length max_response_len
                # 
                # Problem: Different turns have different response lengths. compute_log_prob uses:
                #   response_length = responses.size(1)  # = max_response_len after padding
                #   logits = output[:, -response_length-1:-1]  # slices last response_length logits
                # 
                # If we only left-pad input_ids without restructuring:
                #   Turn 0 (resp_len=200): [PAD...][prompt][resp_200] → slice [-2001:-1] gets prompt! ✗
                #   Turn 3 (resp_len=800): [PAD...][prompt][resp_800] → slice [-2001:-1] gets prompt! ✗
                # 
                # Solution: Reconstruct input_ids to have uniform response length:
                #   1. Extract prompt = input_ids[:, :-original_resp_len]
                #   2. Extract response = input_ids[:, -original_resp_len:]
                #   3. Pad response to max_response_len (right padding)
                #   4. Rebuild input_ids = [prompt][response_padded]
                #   5. Then left-pad the whole input_ids to max_seq_len
                # 
                # After restructuring:
                #   Turn 0: [PAD...][prompt][resp_200][PAD_to_2000] → slice [-2001:-1] gets exactly response ✓
                #   Turn 3: [PAD...][prompt][resp_800][PAD_to_2000] → slice [-2001:-1] gets exactly response ✓
                
                gt_idx = vectorized_data_collector['gt_idx']
                
                # Step 1: Collect all turns' data
                # Note: pseudo_generate_sequences pads responses to the max GT length in the batch.
                # Since pseudo_resps_with_gt is fixed across all turns, response width is uniform.
                all_input_ids = []
                all_attention_mask = []
                all_position_ids = []
                all_responses = []
                turn_batch_sizes = []
                turn_response_lengths = []
                
                for turn_idx, pseudo_output in enumerate(vectorized_data_collector['pseudo_outputs_per_turn']):
                    batch_size = pseudo_output.batch['input_ids'].shape[0]
                    turn_batch_sizes.append(batch_size)
                    all_input_ids.append(pseudo_output.batch['input_ids'])
                    all_attention_mask.append(pseudo_output.batch['attention_mask'])
                    all_position_ids.append(pseudo_output.batch['position_ids'])
                    all_responses.append(pseudo_output.batch['responses'])
                    turn_response_lengths.append(pseudo_output.batch['responses'].shape[1])
                
                max_response_len = max(turn_response_lengths)
                max_seq_len = max(t.shape[1] for t in all_input_ids)
                
                # Step 2: Apply LEFT padding to input_ids, attention_mask, and position_ids
                # CRITICAL: Use original position_ids and apply left padding correctly
                # position_ids for PAD tokens should be 0 (will be ignored due to attention_mask=0)
                padded_input_ids = []
                padded_attention_mask = []
                padded_position_ids = []
                padded_responses = []
                
                for i in range(len(all_input_ids)):
                    curr_seq_len = all_input_ids[i].shape[1]
                    curr_resp_len = turn_response_lengths[i]
                    left_pad_len = max_seq_len - curr_seq_len
                    
                    if left_pad_len > 0:
                        # Left padding for input_ids
                        padded_input_ids.append(F.pad(all_input_ids[i], (left_pad_len, 0), value=self.tokenizer.pad_token_id))
                        # Left padding for attention_mask (PAD = 0)
                        padded_attention_mask.append(F.pad(all_attention_mask[i], (left_pad_len, 0), value=0))
                        # Left padding for position_ids (PAD = 0, will be ignored)
                        padded_position_ids.append(F.pad(all_position_ids[i], (left_pad_len, 0), value=0))
                    else:
                        padded_input_ids.append(all_input_ids[i])
                        padded_attention_mask.append(all_attention_mask[i])
                        padded_position_ids.append(all_position_ids[i])
                    
                    # Responses: right padding to max_response_len (if needed)
                    resp_pad_len = max_response_len - curr_resp_len
                    if resp_pad_len > 0:
                        padded_responses.append(F.pad(all_responses[i], (0, resp_pad_len), value=self.tokenizer.pad_token_id))
                    else:
                        padded_responses.append(all_responses[i])
                
                # Step 3: Merge into one large batch
                merged_input_ids = torch.cat(padded_input_ids, dim=0)
                merged_attention_mask = torch.cat(padded_attention_mask, dim=0)
                merged_position_ids = torch.cat(padded_position_ids, dim=0)
                merged_responses = torch.cat(padded_responses, dim=0)
                
                # Compute prompts (everything except the last max_response_len tokens)
                prompt_len = max_seq_len - max_response_len
                merged_prompts = merged_input_ids[:, :prompt_len]
                
                merged_batch = DataProto.from_dict({
                    'prompts': merged_prompts,
                    'responses': merged_responses,
                    'input_ids': merged_input_ids,
                    'attention_mask': merged_attention_mask,
                    'position_ids': merged_position_ids,
                })
                
                total_samples = merged_input_ids.shape[0]
                print(f"[IGPO] Vectorized: Merged {num_turns_collected} turns, batch={total_samples}, seq_len={max_seq_len}")
                
                # Call compute_log_prob ONCE for all turns
                merged_log_probs = self.actor_rollout_wg.compute_log_prob(merged_batch)
                merged_old_log_probs = merged_log_probs.batch['old_log_probs']
                merged_entropys = merged_log_probs.batch['entropys']
                
                print(f"[IGPO] Vectorized: compute_log_prob completed, merged_old_log_probs.shape = {merged_old_log_probs.shape}")
                
                # Extract each turn's results and compute info_gain_rewards
                # turn_boundaries marks the start index of each turn in the merged batch
                turn_boundaries = [0]
                for bs in turn_batch_sizes:
                    turn_boundaries.append(turn_boundaries[-1] + bs)
                
                for turn_idx in range(num_turns_collected):
                    start_idx = turn_boundaries[turn_idx]
                    end_idx = turn_boundaries[turn_idx + 1]
                    activate_list_for_turn = vectorized_data_collector['activate_lists_per_turn'][turn_idx]
                    
                    # Extract current turn's log_probs from merged results
                    turn_old_log_probs = merged_old_log_probs[start_idx:end_idx]
                    turn_entropys = merged_entropys[start_idx:end_idx]
                    
                    if turn_idx == 0:
                        # First turn: initialize gt_values
                        for local_idx, global_idx in enumerate(activate_list_for_turn):
                            if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                                continue
                            log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                            mean_log_prob = log_probs.mean().item()
                            
                            if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                                continue
                            
                            if info_gain_type == "log_prob_diff":
                                gt_values[global_idx] = mean_log_prob
                            else:
                                gt_values[global_idx] = torch.exp(torch.tensor(mean_log_prob)).item()
                            
                            gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                            gt_entropys_per_turn[global_idx].append(turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist())
                    else:
                        # Subsequent turns: compute info_gain
                        for local_idx, global_idx in enumerate(activate_list_for_turn):
                            if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                                continue
                            if global_idx not in gt_values:
                                continue
                            log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                            mean_log_prob = log_probs.mean().item()
                            
                            if math.isnan(mean_log_prob):
                                continue
                            
                            if info_gain_type == "log_prob_diff":
                                cur_value = mean_log_prob
                                info_gain = cur_value - gt_values[global_idx]
                            else:
                                cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                                info_gain = cur_value - gt_values[global_idx]
                            
                            if math.isnan(info_gain) or math.isinf(info_gain):
                                continue
                            
                            info_gain_rewards[global_idx].append(info_gain)
                            gt_values[global_idx] = cur_value
                            
                            gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                            gt_entropys_per_turn[global_idx].append(turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist())
                
                # Statistics and print results
                total_info_gains = sum(len(r) for r in info_gain_rewards)
                print(f"[IGPO] Vectorized GT LogProb COMPLETED: "
                      f"{num_turns_collected} turns merged, "
                      f"{total_samples} total samples, "
                      f"1 compute_log_prob call, "
                      f"{total_info_gains} info_gain values computed")
                
                # ========== Verification: compare vectorized vs original mode ==========
                if VERIFY_VECTORIZED and original_mode_info_gains is not None:
                    print(f"\n[VERIFY] ========== Comparing Vectorized vs Original Mode ==========")
                    mismatch_count = 0
                    match_count = 0
                    total_compared = 0
                    max_abs_diff = 0.0
                    
                    for sample_idx in range(len(messages_list)):
                        vec_gains = info_gain_rewards[sample_idx]
                        orig_gains = original_mode_info_gains[sample_idx]
                        
                        if len(vec_gains) != len(orig_gains):
                            print(f"[VERIFY] Sample {sample_idx}: LENGTH MISMATCH! vectorized={len(vec_gains)}, original={len(orig_gains)}")
                            mismatch_count += 1
                            continue
                        
                        for turn_idx, (v, o) in enumerate(zip(vec_gains, orig_gains)):
                            total_compared += 1
                            abs_diff = abs(v - o)
                            max_abs_diff = max(max_abs_diff, abs_diff)
                            
                            # Check if values are close (tolerance for floating point)
                            if abs_diff > 1e-5:
                                # Only print first 10 mismatches to avoid log spam
                                if mismatch_count < 10:
                                    print(f"[VERIFY] Sample {sample_idx}, Turn {turn_idx}: MISMATCH! "
                                          f"vectorized={v:.8f}, original={o:.8f}, diff={abs_diff:.8e}")
                                mismatch_count += 1
                            else:
                                match_count += 1
                    
                    print(f"[VERIFY] ========== Summary ==========")
                    print(f"[VERIFY] Total values compared: {total_compared}")
                    print(f"[VERIFY] Matches (diff < 1e-5): {match_count}")
                    print(f"[VERIFY] Mismatches: {mismatch_count}")
                    print(f"[VERIFY] Max absolute difference: {max_abs_diff:.8e}")
                    if mismatch_count == 0 and total_compared > 0:
                        print(f"[VERIFY] ✓ PASSED: Vectorized mode is numerically equivalent to original mode!")
                    elif total_compared == 0:
                        print(f"[VERIFY] ⚠ WARNING: No values to compare (no info_gains computed)")
                    else:
                        print(f"[VERIFY] ✗ FAILED: Found {mismatch_count} mismatches!")
                    print(f"[VERIFY] ====================================\n")
            else:
                print(f"[IGPO] Vectorized GT LogProb: No turns collected (all samples may have finished early)")
        
        # Save gt_log_probs to local output directory (uncomment for debugging if needed)
        # gt_log_probs_path = os.path.join(output_dir, f"gt_log_probs_{global_steps}.json")
        # with open(gt_log_probs_path, 'w') as f:
        #     json.dump({"gt_log_probs_per_turn": gt_log_probs_per_turn, "gt_entropys_per_turn": gt_entropys_per_turn}, f)

        if activate_list != []:
            for i in activate_list:
                # Use add_generation_prompt=False to avoid adding an extra separator at the end,
                # which would cause turn mismatch in info_gain.py
                message_string_list[i] = self.tokenizer.apply_chat_template(messages_list[i], add_generation_prompt=False, tokenize=False)
        
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

        _tokenized_responses = self.tokenizer(response_str_list, return_tensors="pt", padding=True)
        responses = _tokenized_responses['input_ids']
        responses_attention_mask = _tokenized_responses['attention_mask']
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
        print("generation completed")

        print(f"node {node_rank} message_string_list {len(message_string_list)}")

        return message_string_list, message_tensor, info_gain_rewards
