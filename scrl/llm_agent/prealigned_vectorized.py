"""
IGPO Prealigned Vectorized GT LogProb Computation

This module implements the prealigned prompt vectorization strategy for GT LogProb computation.
It is completely independent and does not affect the original computation mode.

Key Design Principles:
1. Complete decoupling: This module only processes collected data, never modifies original data flow
2. Mathematical rigor: Prealigned prompts ensure response position_ids are identical to original mode
3. Strict validation: Built-in checkpoints to verify results match original mode exactly
4. Minimal footprint: Only called when vectorized mode is enabled, zero impact otherwise

Usage:
    from scrl.llm_agent.prealigned_vectorized import compute_vectorized_gt_logprob
    
    # Only call after collecting all turns' data
    results = compute_vectorized_gt_logprob(
        pseudo_outputs_per_turn=collected_outputs,
        activate_lists_per_turn=collected_activate_lists,
        gt_idx=gt_idx,
        actor_rollout_wg=self.actor_rollout_wg,
        tokenizer=self.tokenizer,
        info_gain_type=self.config.info_gain_type,
        enable_strict_validation=True,  # Set to True for debugging
    )
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import copy

# Try to import DataProto, handle if not available
try:
    from verl.protocol import DataProto
except ImportError:
    DataProto = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PrealignedVectorizedConfig:
    """Configuration for prealigned vectorized computation."""
    # Validation settings
    enable_strict_validation: bool = False  # Enable validation against original mode
    validation_tolerance: float = 1e-6      # Tolerance for numerical comparison
    # Debug settings
    debug_print: bool = False               # Print debug information
    debug_samples: List[int] = None         # Specific samples to track


def get_config_from_env() -> PrealignedVectorizedConfig:
    """Get configuration from environment variables."""
    return PrealignedVectorizedConfig(
        enable_strict_validation=os.environ.get('IGPO_VECTORIZED_STRICT_VALIDATION', '').lower() in ('true', '1'),
        debug_print=os.environ.get('IGPO_VECTORIZED_DEBUG', '').lower() in ('true', '1'),
    )


# ============================================================================
# Core Functions: Prealigned Prompt Processing
# ============================================================================

def prealign_single_turn(
    pseudo_output: Any,  # DataProto
    target_prompt_len: int,
    pad_token_id: int,
) -> Any:  # DataProto
    """
    Prealign a single turn's pseudo_output to target_prompt_len.
    
    This function applies RIGHT padding to the prompt part, keeping the response
    at the end of the sequence. This ensures response's position_ids are identical
    to the original mode.
    
    Original format:  [Prompt_actual][Response]
    Prealigned format: [Prompt_actual][PAD...][Response]
                                      ↑ right padding
    
    Args:
        pseudo_output: DataProto containing prompts, responses, input_ids, etc.
        target_prompt_len: Target length for prompt (including padding)
        pad_token_id: Token ID for padding
        
    Returns:
        New DataProto with prealigned data (original is not modified)
    """
    # Extract original tensors
    prompts = pseudo_output.batch['prompts']           # (N, actual_prompt_len)
    responses = pseudo_output.batch['responses']       # (N, response_len)
    input_ids = pseudo_output.batch['input_ids']       # (N, actual_seq_len)
    attention_mask = pseudo_output.batch['attention_mask']
    position_ids = pseudo_output.batch['position_ids']
    
    batch_size = prompts.shape[0]
    actual_prompt_len = prompts.shape[1]
    response_len = responses.shape[1]
    actual_seq_len = input_ids.shape[1]
    device = input_ids.device
    
    # Calculate padding needed
    pad_len = target_prompt_len - actual_prompt_len
    
    if pad_len <= 0:
        # No padding needed, return a clone to avoid modifying original
        return DataProto.from_dict({
            'prompts': prompts.clone(),
            'responses': responses.clone(),
            'input_ids': input_ids.clone(),
            'attention_mask': attention_mask.clone(),
            'position_ids': position_ids.clone(),
        })
    
    # ========== Step 1: Prealign prompts (right padding) ==========
    aligned_prompts = F.pad(prompts, (0, pad_len), value=pad_token_id)
    # Shape: (N, target_prompt_len)
    
    # ========== Step 2: Rebuild input_ids ==========
    # New format: [aligned_prompts][responses]
    aligned_input_ids = torch.cat([aligned_prompts, responses], dim=1)
    # Shape: (N, target_prompt_len + response_len)
    
    # ========== Step 3: Rebuild attention_mask ==========
    # Original prompt mask + PAD mask (0) + original response mask
    prompt_mask = attention_mask[:, :actual_prompt_len]
    pad_mask = torch.zeros(batch_size, pad_len, dtype=attention_mask.dtype, device=device)
    response_mask = attention_mask[:, actual_prompt_len:]  # Original response mask
    aligned_attention_mask = torch.cat([prompt_mask, pad_mask, response_mask], dim=1)
    
    # ========== Step 4: Rebuild position_ids ==========
    # CRITICAL: Keep response's position_ids unchanged from original!
    # This is the key to mathematical equivalence.
    #
    # Original:   [0, 1, ..., k, k+1, ..., k+m]
    #              ↑ prompt ↑   ↑ response ↑
    #
    # Prealigned: [0, 1, ..., k, ?, ?, ..., ?, k+1, ..., k+m]
    #              ↑ prompt ↑  ↑ PAD (don't care) ↑  ↑ response (unchanged!) ↑
    #
    prompt_pos = position_ids[:, :actual_prompt_len]
    pad_pos = torch.zeros(batch_size, pad_len, dtype=position_ids.dtype, device=device)
    response_pos = position_ids[:, actual_prompt_len:]  # Keep original response position_ids!
    aligned_position_ids = torch.cat([prompt_pos, pad_pos, response_pos], dim=1)
    
    return DataProto.from_dict({
        'prompts': aligned_prompts,
        'responses': responses.clone(),  # Response unchanged
        'input_ids': aligned_input_ids,
        'attention_mask': aligned_attention_mask,
        'position_ids': aligned_position_ids,
    })


def merge_prealigned_turns(
    aligned_outputs: List[Any],  # List[DataProto]
) -> Any:  # DataProto
    """
    Merge all prealigned turns into a single batch.
    
    Since all turns are prealigned to the same seq_len, we can simply concatenate.
    
    Args:
        aligned_outputs: List of prealigned DataProto objects
        
    Returns:
        Merged DataProto with all turns concatenated
    """
    merged_input_ids = torch.cat([o.batch['input_ids'] for o in aligned_outputs], dim=0)
    merged_attention_mask = torch.cat([o.batch['attention_mask'] for o in aligned_outputs], dim=0)
    merged_position_ids = torch.cat([o.batch['position_ids'] for o in aligned_outputs], dim=0)
    merged_responses = torch.cat([o.batch['responses'] for o in aligned_outputs], dim=0)
    merged_prompts = torch.cat([o.batch['prompts'] for o in aligned_outputs], dim=0)
    
    return DataProto.from_dict({
        'prompts': merged_prompts,
        'responses': merged_responses,
        'input_ids': merged_input_ids,
        'attention_mask': merged_attention_mask,
        'position_ids': merged_position_ids,
    })


# ============================================================================
# Core Function: Main Entry Point
# ============================================================================

def compute_vectorized_gt_logprob(
    pseudo_outputs_per_turn: List[Any],      # List[DataProto]
    activate_lists_per_turn: List[List[int]],
    gt_idx: List[List[int]],
    actor_rollout_wg: Any,
    tokenizer: Any,
    info_gain_type: str = "prob_diff",
    enable_strict_validation: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for prealigned vectorized GT LogProb computation.
    
    This function:
    1. Prealigns all turns' prompts to the same length
    2. Merges all turns into a single batch
    3. Calls compute_log_prob ONCE
    4. Extracts results and computes info_gain
    5. (Optional) Validates results against original mode
    
    Args:
        pseudo_outputs_per_turn: List of pseudo_gen_output for each turn (ORIGINAL format, not modified)
        activate_lists_per_turn: List of activate_list for each turn
        gt_idx: GT token range for each sample [(start, end), ...]
        actor_rollout_wg: Actor worker group for compute_log_prob
        tokenizer: Tokenizer for pad_token_id
        info_gain_type: "prob_diff" or "log_prob_diff"
        enable_strict_validation: If True, also compute using original mode and compare
        original_mode_compute_fn: Function to compute original mode results (for validation)
        
    Returns:
        Dictionary containing:
        - gt_values: Final gt_values dictionary
        - info_gain_rewards: List of info_gain rewards per sample
        - gt_log_probs_per_turn: Log probs per turn per sample
        - gt_entropys_per_turn: Entropys per turn per sample
        - validation_passed: (if validation enabled) Whether validation passed
        - validation_details: (if validation enabled) Detailed comparison results
    """
    num_turns = len(pseudo_outputs_per_turn)
    if num_turns == 0:
        return {
            'gt_values': {},
            'info_gain_rewards': [],
            'gt_log_probs_per_turn': [],
            'gt_entropys_per_turn': [],
        }
    
    num_samples = pseudo_outputs_per_turn[0].batch['input_ids'].shape[0]
    pad_token_id = tokenizer.pad_token_id
    
    print(f"[PREALIGNED VECTORIZED] Starting: {num_turns} turns, {num_samples} samples/turn")
    
    # ========== Step 1: Determine max_prompt_len ==========
    # Find the maximum prompt length across all turns
    max_prompt_len = 0
    response_len = pseudo_outputs_per_turn[0].batch['responses'].shape[1]
    
    for pseudo_output in pseudo_outputs_per_turn:
        prompt_len = pseudo_output.batch['prompts'].shape[1]
        max_prompt_len = max(max_prompt_len, prompt_len)
    
    max_seq_len = max_prompt_len + response_len
    print(f"[PREALIGNED VECTORIZED] max_prompt_len={max_prompt_len}, response_len={response_len}, max_seq_len={max_seq_len}")
    
    # ========== Step 2: Prealign all turns ==========
    aligned_outputs = []
    for turn_idx, pseudo_output in enumerate(pseudo_outputs_per_turn):
        aligned_output = prealign_single_turn(
            pseudo_output=pseudo_output,
            target_prompt_len=max_prompt_len,
            pad_token_id=pad_token_id,
        )
        aligned_outputs.append(aligned_output)
        
        # Debug: Print first few turns' info
        if turn_idx < 3:
            orig_seq_len = pseudo_output.batch['input_ids'].shape[1]
            aligned_seq_len = aligned_output.batch['input_ids'].shape[1]
            print(f"[PREALIGNED] Turn {turn_idx}: original_seq_len={orig_seq_len}, aligned_seq_len={aligned_seq_len}")
    
    # ========== Step 3: Merge all turns ==========
    merged_batch = merge_prealigned_turns(aligned_outputs)
    total_batch_size = merged_batch.batch['input_ids'].shape[0]
    
    print(f"[PREALIGNED VECTORIZED] Merged: {total_batch_size} total samples (= {num_turns} turns × {num_samples})")
    
    # ========== Step 4: Call compute_log_prob ONCE ==========
    print(f"[PREALIGNED VECTORIZED] Calling compute_log_prob ONCE...")
    merged_log_probs_result = actor_rollout_wg.compute_log_prob(merged_batch)
    merged_old_log_probs = merged_log_probs_result.batch['old_log_probs']
    merged_entropys = merged_log_probs_result.batch['entropys']
    
    print(f"[PREALIGNED VECTORIZED] compute_log_prob completed, shape={merged_old_log_probs.shape}")
    
    # ========== Step 5: Extract results per turn ==========
    gt_values = {}
    info_gain_rewards = [[] for _ in range(num_samples)]
    gt_log_probs_per_turn = [[] for _ in range(num_samples)]
    gt_entropys_per_turn = [[] for _ in range(num_samples)]
    
    # For validation: store mean_log_probs per turn per sample
    vectorized_mean_log_probs = [[] for _ in range(num_samples)]
    
    for turn_idx in range(num_turns):
        start_idx = turn_idx * num_samples
        end_idx = (turn_idx + 1) * num_samples
        
        turn_old_log_probs = merged_old_log_probs[start_idx:end_idx]
        turn_entropys = merged_entropys[start_idx:end_idx]
        activate_list = activate_lists_per_turn[turn_idx]
        
        if turn_idx == 0:
            # First turn: initialize gt_values
            for global_idx in activate_list:
                if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                    continue
                
                log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                mean_log_prob = log_probs.mean().item()
                
                if math.isnan(mean_log_prob) or math.isinf(mean_log_prob):
                    continue
                
                # Store for validation
                vectorized_mean_log_probs[global_idx].append(mean_log_prob)
                
                if info_gain_type == "log_prob_diff":
                    gt_values[global_idx] = mean_log_prob
                else:
                    gt_values[global_idx] = torch.exp(torch.tensor(mean_log_prob)).item()
                
                gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                gt_entropys_per_turn[global_idx].append(
                    turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist()
                )
        else:
            # Subsequent turns: compute info_gain
            for global_idx in activate_list:
                if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                    continue
                if global_idx not in gt_values:
                    continue
                
                log_probs = turn_old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
                mean_log_prob = log_probs.mean().item()
                
                if math.isnan(mean_log_prob):
                    info_gain_rewards[global_idx].append(0.0)
                    continue
                
                # Store for validation
                vectorized_mean_log_probs[global_idx].append(mean_log_prob)
                
                if info_gain_type == "log_prob_diff":
                    cur_value = mean_log_prob
                    info_gain = cur_value - gt_values[global_idx]
                else:
                    cur_value = torch.exp(torch.tensor(mean_log_prob)).item()
                    info_gain = cur_value - gt_values[global_idx]
                
                if math.isnan(info_gain) or math.isinf(info_gain):
                    info_gain_rewards[global_idx].append(0.0)
                    continue
                
                info_gain_rewards[global_idx].append(info_gain)
                gt_values[global_idx] = cur_value
                
                gt_log_probs_per_turn[global_idx].append(log_probs.tolist())
                gt_entropys_per_turn[global_idx].append(
                    turn_entropys[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]].tolist()
                )
    
    # Statistics
    total_info_gains = sum(len(r) for r in info_gain_rewards)
    print(f"[PREALIGNED VECTORIZED] COMPLETED: {num_turns} turns, {total_info_gains} info_gains, 1 compute_log_prob call")
    
    result = {
        'gt_values': gt_values,
        'info_gain_rewards': info_gain_rewards,
        'gt_log_probs_per_turn': gt_log_probs_per_turn,
        'gt_entropys_per_turn': gt_entropys_per_turn,
        'vectorized_mean_log_probs': vectorized_mean_log_probs,
    }
    
    # ========== Step 6: Strict Validation (if enabled) ==========
    # When enabled, compute original mode results (T calls to compute_log_prob) and compare
    # with vectorized results to ensure mathematical equivalence.
    if enable_strict_validation:
        print(f"[PREALIGNED VECTORIZED] Running strict validation...")
        print(f"[PREALIGNED VECTORIZED] This will make {num_turns} additional compute_log_prob calls for verification.")
        validation_result = _run_strict_validation(
            pseudo_outputs_per_turn=pseudo_outputs_per_turn,
            activate_lists_per_turn=activate_lists_per_turn,
            gt_idx=gt_idx,
            actor_rollout_wg=actor_rollout_wg,
            info_gain_type=info_gain_type,
            vectorized_mean_log_probs=vectorized_mean_log_probs,
        )
        result['validation_passed'] = validation_result['passed']
        result['validation_details'] = validation_result['details']
        
        if validation_result['passed']:
            print(f"[PREALIGNED VECTORIZED] ✓ Validation PASSED! Max diff: {validation_result['max_diff']:.2e}")
        else:
            print(f"[PREALIGNED VECTORIZED] ✗ Validation FAILED! Max diff: {validation_result['max_diff']:.2e}")
            print(f"[PREALIGNED VECTORIZED] Details: {validation_result['details']}")
    
    return result


# ============================================================================
# Validation Functions
# ============================================================================

def _run_strict_validation(
    pseudo_outputs_per_turn: List[Any],
    activate_lists_per_turn: List[List[int]],
    gt_idx: List[List[int]],
    actor_rollout_wg: Any,
    info_gain_type: str,
    vectorized_mean_log_probs: List[List[float]],
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Run strict validation by computing original mode results and comparing.
    
    This function computes GT LogProb using the original mode (one compute_log_prob
    call per turn) and compares with vectorized results.
    
    Args:
        pseudo_outputs_per_turn: Original format pseudo_outputs (not prealigned)
        activate_lists_per_turn: Activate lists per turn
        gt_idx: GT token ranges
        actor_rollout_wg: Actor worker group
        info_gain_type: "prob_diff" or "log_prob_diff"
        vectorized_mean_log_probs: Mean log probs from vectorized computation
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with validation results
    """
    num_turns = len(pseudo_outputs_per_turn)
    num_samples = len(vectorized_mean_log_probs)
    
    # Compute original mode results
    original_mean_log_probs = [[] for _ in range(num_samples)]
    
    print(f"[VALIDATION] Computing original mode results for {num_turns} turns...")
    
    for turn_idx, pseudo_output in enumerate(pseudo_outputs_per_turn):
        # Call compute_log_prob for this turn (original mode)
        log_probs_result = actor_rollout_wg.compute_log_prob(pseudo_output)
        old_log_probs = log_probs_result.batch['old_log_probs']
        
        activate_list = activate_lists_per_turn[turn_idx]
        
        for global_idx in activate_list:
            if gt_idx[global_idx][0] >= gt_idx[global_idx][1]:
                continue
            
            log_probs = old_log_probs[global_idx, gt_idx[global_idx][0]:gt_idx[global_idx][1]]
            mean_log_prob = log_probs.mean().item()
            
            if not math.isnan(mean_log_prob) and not math.isinf(mean_log_prob):
                original_mean_log_probs[global_idx].append(mean_log_prob)
    
    # Compare results
    total_compared = 0
    total_matched = 0
    total_mismatched = 0
    max_diff = 0.0
    mismatch_details = []
    
    for sample_idx in range(num_samples):
        vec_probs = vectorized_mean_log_probs[sample_idx]
        orig_probs = original_mean_log_probs[sample_idx]
        
        # Check length match
        if len(vec_probs) != len(orig_probs):
            mismatch_details.append({
                'type': 'length_mismatch',
                'sample': sample_idx,
                'vec_len': len(vec_probs),
                'orig_len': len(orig_probs),
            })
            continue
        
        # Compare values
        for turn_idx, (v, o) in enumerate(zip(vec_probs, orig_probs)):
            total_compared += 1
            diff = abs(v - o)
            max_diff = max(max_diff, diff)
            
            if diff <= tolerance:
                total_matched += 1
            else:
                total_mismatched += 1
                if len(mismatch_details) < 10:  # Limit details
                    mismatch_details.append({
                        'type': 'value_mismatch',
                        'sample': sample_idx,
                        'turn': turn_idx,
                        'vectorized': v,
                        'original': o,
                        'diff': diff,
                    })
    
    passed = (total_mismatched == 0) and (total_compared > 0)
    
    print(f"[VALIDATION] Compared {total_compared} values: {total_matched} matched, {total_mismatched} mismatched")
    print(f"[VALIDATION] Max absolute difference: {max_diff:.2e}")
    
    return {
        'passed': passed,
        'total_compared': total_compared,
        'total_matched': total_matched,
        'total_mismatched': total_mismatched,
        'max_diff': max_diff,
        'details': mismatch_details,
    }


def validate_prealignment_correctness(
    original_output: Any,  # DataProto
    aligned_output: Any,   # DataProto
    sample_idx: int = 0,
) -> Dict[str, Any]:
    """
    Validate that prealignment preserves response position_ids correctly.
    
    This is a debugging utility to verify the prealignment logic.
    
    Args:
        original_output: Original pseudo_gen_output
        aligned_output: Prealigned pseudo_gen_output
        sample_idx: Which sample to check
        
    Returns:
        Dictionary with validation results
    """
    orig_pos = original_output.batch['position_ids'][sample_idx]
    aligned_pos = aligned_output.batch['position_ids'][sample_idx]
    
    orig_prompt_len = original_output.batch['prompts'].shape[1]
    aligned_prompt_len = aligned_output.batch['prompts'].shape[1]
    
    # Extract response position_ids
    orig_response_pos = orig_pos[orig_prompt_len:].tolist()
    aligned_response_pos = aligned_pos[aligned_prompt_len:].tolist()
    
    # Check if response position_ids are identical
    response_pos_match = (orig_response_pos == aligned_response_pos)
    
    return {
        'response_position_ids_match': response_pos_match,
        'original_prompt_len': orig_prompt_len,
        'aligned_prompt_len': aligned_prompt_len,
        'original_response_pos': orig_response_pos[:10],  # First 10 for display
        'aligned_response_pos': aligned_response_pos[:10],
    }
