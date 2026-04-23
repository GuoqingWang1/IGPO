<h1 align="center">✨ Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents

</h1>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2510.14967)
[![Paper](https://img.shields.io/badge/🤗%20Paper-Hugging%20Face-yellow)](https://huggingface.co/papers/2510.14967)
[![Ant Group](https://img.shields.io/badge/Supported%20by-Ant%20Group-1677FF.svg?logo=ant-design)](https://www.antgroup.com/)

</div>

## 📣 Latest News
- **[Apr 21, 2026]**: We have successfully extended **IGPO** to handle ultra-long-horizon Deep Research tasks (200+ turns). Based on this, we trained **DR-Venus**, a frontier 4B deep research agent for edge-scale deployment. Empowered by **IGPO**, the performance of **DR-Venus** is effectively improved on challenging benchmarks, including **BrowseComp** and **BrowseComp-ZH**. For more details, please refer to the DR-Venus technical report [![DR-Venus](https://img.shields.io/badge/DR--Venus-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/pdf/2604.19859) [![DR-Venus](https://img.shields.io/badge/🤗%20DR--Venus-Hugging%20Face-yellow)](https://huggingface.co/papers/2604.19859), the released models [![DR-Venus-4B-RL](https://img.shields.io/badge/🤗%20DR--Venus--4B--RL-Hugging%20Face-yellow)](https://huggingface.co/inclusionAI/DR-Venus-4B-RL), and the IGPO training codebase within the DR-Venus codebase [![GitHub](https://img.shields.io/badge/Github-DR--Venus%2FRL-181717?logo=github&logoColor=white)](https://github.com/inclusionAI/DR-Venus/tree/master/RL).
- **[Feb 01, 2026]**: 🔄 Codebase updated with new features. [See details](resources/CHANGELOG_20260201.md)
- **[Jan 26, 2026]**: 🎉 Our **[IGPO](https://arxiv.org/abs/2510.14967)** paper has been accepted at **ICLR 2026**!
- **[Oct 17, 2025]**: 📄 Our IGPO paper is now available on **[arXiv](https://arxiv.org/abs/2510.14967)** and **[Hugging Face](https://huggingface.co/papers/2510.14967)** daily paper.

## 🤖Models
[![DR-Venus-4B-RL](https://img.shields.io/badge/🤗%20DR--Venus--4B--RL-Hugging%20Face-yellow)](https://huggingface.co/inclusionAI/DR-Venus-4B-RL) [![DR-Venus-4B-RL-GGUF](https://img.shields.io/badge/🤗%20DR--Venus--4B--RL--GGUF-Hugging%20Face-yellow)](https://huggingface.co/inclusionAI/DR-Venus-4B-RL-GGUF)

## 💡 Method Overview
We introduce IGPO, a RL algorithm for fine-grained credit assignment in search agent training. By modeling agentic search turns as an incremental information acquisition process, IGPO defines rewards as the marginal gain in the policy's probability of generating the correct answer. 

<p align="center">
    <img src="./images/Framework.png" width="100%">
</p>

## 📊 Overall Performance
<p align="center">
    <img src="./images/Exp.png" width="100%">
</p>

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/GuoqingWang1/IGPO
cd IGPO

conda create -n igpo python=3.10
conda activate igpo

pip install -r requirements.txt

pip install -e .
```

### 2. Configure Web Search API & Prompt Template

Edit `tools_server/config.yaml`:

```yaml
# Google Search (via Serper API)
search_engine: "google"
serper_api_key: "your_serper_api_key_here"

# Or Bing Search (via Azure)
# search_engine: "bing"
# azure_bing_search_subscription_key: "your_bing_key_here"

# System prompt template is also defined in this file
# system_prompt: |-
#   ...
```

> **💡 Tip**: If you don't have access to a search API yet, you can use **mock mode** for testing:
> ```bash
> export IGPO_MOCK_SEARCH=true
> ```
> Or set `mock_mode: true` in `tools_server/config.yaml`. This will return simulated search results without actual API calls.

### 3. Prepare Data

Place your training data in the `data/` directory:
- `data/train.parquet` - Training data
- `data/dev.parquet` - Validation data
- `data/test.parquet` - Evaluation data

Data format: See the provided data for reference.

### 4. Training

> **Supported Models**: We currently support Qwen series models (e.g., Qwen2.5-7B-Instruct).

Edit `train.sh` to configure training parameters, then run:

```bash
bash train.sh
```

<details>
<summary><b>Key Parameters in <code>train.sh</code></b></summary> 

| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to your model or Hugging Face model name (e.g., `Qwen/Qwen2.5-7B-Instruct`) |
| `OUTPUT` | Directory for saving checkpoints |
| `EVAL_LOG_PATH` | Directory for saving validation results |
| `data.train_files` | Path to training data (parquet format) |
| `data.val_files` | Path to validation data (parquet format) |
| `algorithm.gamma` | Discount factor for reward computation |
| `+algorithm.info_gain_type` | Info gain reward calculation: `log_prob_diff` (log probability difference) or `prob_diff` (probability difference) |
| `+algorithm.info_gain_norm_mode` | Reward normalization: `separate` (normalize info gain and outcome rewards independently) or `joint` (normalize all rewards together) |
| `+algorithm.use_vectorized_gt_logprob` | Enable vectorized ground truth log probability computation for faster info gain reward construction |
| `+algorithm.use_curriculum` | Enable curriculum learning to gradually decay info gain reward weight during training |
| `+algorithm.curriculum_f1_init/final` | Initial and final weight for outcome reward in curriculum learning |
| `+algorithm.curriculum_ig_init/final` | Initial and final weight for info gain reward in curriculum learning |
| `trainer.save_freq` | Save checkpoint every N training steps |
| `trainer.test_freq` | Run validation every N training steps |
| `agent_grpo.n` | Number of rollouts per sample (GRPO group size) |
| `max_turns` | Maximum number of search turns allowed per episode |

</details>

### 5. Evaluation

Edit `evaluate.sh` to configure evaluation parameters, then run:

```bash
bash evaluate.sh
```

<details>
<summary><b>Key Parameters in <code>evaluate.sh</code></b></summary>

| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to your trained checkpoint |
| `TEST_FILES` | Path to test data (parquet format) |
| `OUTPUT_DIR` | Directory for saving evaluation results |
| `EVAL_LOG_PATH` | Directory for saving evaluation logs |
| `MAX_TURNS` | Maximum number of search turns allowed per episode |

</details>

## 📄 Citation
If you find our code or work useful for your research, please cite our work.
```bibtex
@inproceedings{
wang2026information,
title={Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents},
author={Guoqing Wang and Sunhao Dai and Guangze Ye and Zeyu Gan and Wei Yao and Yong Deng and Xiaofeng Wu and Zhenzhe Ying},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=qkWP6phrvZ}
}
```

## 🙏 Acknowledgement 

IGPO is inspired by [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1), with its implementation built upon [veRL](https://github.com/volcengine/verl), [Search-r1](https://github.com/PeterGriffinJin/Search-R1), and [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher). We are grateful to the teams behind these projects for their significant contributions to open-source research and development.

## 📞 Contact

For any questions or feedback, please reach out to us at guoqingwang905@gmail.com.

## 📜 License

This project is released under the [MIT License](LICENSE).

## 🌟 Star History

<a href="https://star-history.com/#GuoqingWang1/IGPO&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=GuoqingWang1/IGPO&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=GuoqingWang1/IGPO&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=GuoqingWang1/IGPO&type=Date" />
 </picture>
</a>
