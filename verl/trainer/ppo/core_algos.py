# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
    turn_boundary_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Turn-level discounted accumulation + broadcast implementation.
    
    Each turn is defined by reward position (non-zero reward marks end of turn).
    
    Computation flow:
    1. Identify turn boundaries for each sample (based on reward positions)
    2. Turn-level discounted accumulation: A_i = r_i + gamma * A_{i+1}
    3. Broadcast: Broadcast A_i to all tokens in turn i
    
    Args:
        normalized_rewards: Normalized rewards (bsz, seq_len)
        response_mask: Response mask (bsz, seq_len)
        gamma: Discount factor
        bsz: batch size
        seq_len: Sequence length
        device: Device
        turn_boundary_mask: Optional pre-computed mask (bsz, seq_len) identifying
            turn boundary positions. When provided, used instead of != 0 heuristic
            to avoid missing boundaries where normalized rewards happen to be zero.
    
    Returns:
        discounted_returns: Turn-level advantage broadcast to all tokens (bsz, seq_len)
    """
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)
    
    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]  # (seq_len,)
        sample_mask = response_mask[sample_idx]  # (seq_len,)
        
        # Step 1: Find all reward positions (turn end positions)
        if turn_boundary_mask is not None:
            reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        else:
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()
        
        if len(reward_positions) == 0:
            # No reward, skip
            continue
        
        # Step 2: Turn-level discounted accumulation (backward)
        # turn_data: [(reward_pos, turn_advantage), ...]
        turn_data = []
        next_turn_adv = 0.0
        
        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv = turn_reward + gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv
        
        turn_data.reverse()  # Convert to forward order
        
        # Step 3: Broadcast to all tokens in each turn
        # Turn i range: [prev_reward_pos + 1, current_reward_pos]
        # First turn starts from position 0
        prev_end = 0
        for i, (reward_pos, adv) in enumerate(turn_data):
            # Turn range: [prev_end, reward_pos]
            # Only broadcast to positions where response_mask == 1
            for t in range(prev_end, reward_pos + 1):
                if sample_mask[t] == 1:
                    discounted_returns[sample_idx, t] = adv
            prev_end = reward_pos + 1
    
    return discounted_returns


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    gamma: float = 1.0,
    info_gain_norm_mode: str = "joint",
    curriculum_f1_weight: float = 1.0,
    curriculum_ig_weight: float = 1.0,
):
    """
    Compute advantage for GRPO using Turn-level accumulation + broadcast.
    
    Computation flow:
    1. Normalize rewards (info_gain and f1)
    2. Turn-level discounted accumulation: A_i = r_i + gamma * A_{i+1}
    3. Broadcast each turn's advantage to all tokens in that turn
    
    Args:
        token_level_rewards: (bs, response_length) Immediate reward for each token
        response_mask: (bs, response_length) Response sequence mask
        index: Prompt index array for grouping samples
        epsilon: Small constant to prevent division by zero
        norm_adv_by_std_in_grpo: Whether to divide by standard deviation
        gamma: Discount factor, default 1.0
        info_gain_norm_mode: "joint" or "separate"
        curriculum_f1_weight: Curriculum weight for F1 reward, default 1.0
        curriculum_ig_weight: Curriculum weight for InfoGain reward, default 1.0

    Returns:
        advantages, returns: Both are (bs, response_length)
    """
    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    # ========== Step 1: Build masks ==========
    with torch.no_grad():
        last_valid_pos = (seq_len - 1) - response_mask.flip(dims=[1]).to(torch.long).argmax(dim=1)
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        f1_mask = (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        ig_mask = (response_mask == 1) & (~f1_mask) & (token_level_rewards != 0)
    
    # ========== Step 1.5: Apply Curriculum weights ==========
    if curriculum_f1_weight != 1.0 or curriculum_ig_weight != 1.0:
        weighted_rewards = token_level_rewards.clone()
        weighted_rewards = torch.where(f1_mask, token_level_rewards * curriculum_f1_weight, weighted_rewards)
        weighted_rewards = torch.where(ig_mask, token_level_rewards * curriculum_ig_weight, weighted_rewards)
        token_level_rewards = weighted_rewards

    # ========== Step 2: Build Group mapping (vectorized) ==========
    # Convert index to consecutive group_id (0, 1, 2, ...)
    unique_indices, inverse_indices = np.unique(index, return_inverse=True)
    group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)  # (bsz,)
    num_groups = len(unique_indices)
    
    # Expand group_ids to (bsz, seq_len)
    group_ids_expanded = group_ids.unsqueeze(1).expand(-1, seq_len)

    # ========== Step 3: Vectorized computation of group statistics ==========
    def compute_group_stats(mask):
        """Compute mean and std for each group at mask positions"""
        flat_mask = mask.view(-1)
        flat_rewards = token_level_rewards.view(-1)
        flat_group_ids = group_ids_expanded.reshape(-1)
        
        # Select only valid positions
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return torch.zeros(num_groups, device=device), torch.ones(num_groups, device=device)
        
        valid_rewards = flat_rewards[valid_idx]
        valid_groups = flat_group_ids[valid_idx]
        
        # Compute sum and count
        group_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, valid_rewards)
        group_count = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, torch.ones_like(valid_rewards))
        
        # Mean
        group_mean = group_sum / group_count.clamp(min=1.0)
        
        # Std: Using E[(x - mean)^2] formula
        expanded_mean = group_mean[valid_groups]
        sq_diff = (valid_rewards - expanded_mean) ** 2
        group_sq_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, sq_diff)
        group_var = group_sq_sum / group_count.clamp(min=1.0)
        group_std = torch.sqrt(group_var + 1e-8)
        
        # When count <= 1, set std to 1.0
        group_std = torch.where(group_count <= 1, torch.ones_like(group_std), group_std)
        
        return group_mean, group_std

    # ========== Step 4: Vectorized normalization ==========
    normalized_rewards = torch.zeros_like(token_level_rewards)

    if info_gain_norm_mode == "separate":
        # F1 part
        f1_mean, f1_std = compute_group_stats(f1_mask)
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map = f1_std[group_ids_expanded]
        
        norm_f1 = (token_level_rewards - f1_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)
        
        # InfoGain part
        ig_mean, ig_std = compute_group_stats(ig_mask)
        ig_mean_map = ig_mean[group_ids_expanded]
        ig_std_map = ig_std[group_ids_expanded]
        
        norm_ig = (token_level_rewards - ig_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_ig = norm_ig / (ig_std_map + epsilon)
        normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)
    
    else:  # joint
        joint_mask = f1_mask | ig_mask
        g_mean, g_std = compute_group_stats(joint_mask)
        mean_map = g_mean[group_ids_expanded]
        std_map = g_std[group_ids_expanded]
        
        norm_val = (token_level_rewards - mean_map)
        if norm_adv_by_std_in_grpo:
            norm_val = norm_val / (std_map + epsilon)
        normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

    # ========== Step 5: Turn-level discounted accumulation + broadcast ==========
    # Each turn's advantage is computed through turn-level discounted accumulation
    # Then broadcast to all tokens in that turn
    # Use f1_mask | ig_mask (computed before normalization) as turn boundaries
    # to avoid missing turns whose normalized reward happens to be zero.
    discounted_returns = _compute_turn_level_advantage(
        normalized_rewards=normalized_rewards,
        response_mask=response_mask,
        gamma=gamma,
        bsz=bsz,
        seq_len=seq_len,
        device=device,
        turn_boundary_mask=f1_mask | ig_mask,
    )

    return discounted_returns, discounted_returns


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
