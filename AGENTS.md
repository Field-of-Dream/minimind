# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-18
**Commit:** dddedc6
**Branch:** master

## OVERVIEW
MiniMind - ultra-small LLM (~64M params) trainable from scratch in 2h/$3. Full training pipeline: pretrain → SFT → RLHF (DPO/PPO/GRPO) → LoRA → distillation → agent RL. All core algorithms in raw PyTorch, no high-level abstractions.

## STRUCTURE
```
minimind/
├── model/           # MiniMind architecture (GPT-style, MoE support, LoRA)
├── dataset/         # PyTorch Datasets (pretrain, SFT, DPO, RLAIF, agent)
├── trainer/         # Training scripts for each pipeline stage
├── scripts/         # Serving, eval, demo, conversion utilities
├── eval_llm.py      # Interactive chat / model evaluation entry point
├── requirements.txt # All deps (torch, transformers, trl, wandb, etc.)
└── images/          # README assets
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model architecture | `model/model_minimind.py` | MiniMindConfig, MiniMindForCausalLM, MoE blocks |
| LoRA implementation | `model/model_lora.py` | LoRA class, apply_lora(), save/load |
| Dataset classes | `dataset/lm_dataset.py` | PretrainDataset, SFTDataset, DPODataset, RLAIFDataset, AgentRLDataset |
| Pretraining | `trainer/train_pretrain.py` | Raw text → next-token prediction |
| SFT | `trainer/train_full_sft.py` | Instruction tuning |
| DPO (RLHF) | `trainer/train_dpo.py` | Direct preference optimization |
| PPO | `trainer/train_ppo.py` | Actor-critic RL with rollout engine |
| GRPO | `trainer/train_grpo.py` | Group-relative policy optimization |
| LoRA training | `trainer/train_lora.py` | Low-rank adaptation |
| Distillation | `trainer/train_distillation.py` | KL-divergence student-teacher |
| Agent RL | `trainer/train_agent.py` | Tool-use / agentic training |
| Rollout engine | `trainer/rollout_engine.py` | Generation for RL training loops |
| Tokenizer training | `trainer/train_tokenizer.py` | BPE tokenizer (not recommended - use provided) |
| OpenAI API server | `scripts/serve_openai_api.py` | FastAPI-compatible endpoint |
| Web demo | `scripts/web_demo.py` | Streamlit UI |
| Chat CLI | `scripts/chat_api.py` | Terminal chat interface |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| MiniMindConfig | class | model/model_minimind.py:10 | Model hyperparameters (hidden_size, layers, MoE, RoPE) |
| MiniMindForCausalLM | class | model/model_minimind.py:48 | Main model: embedding, RMSNorm, attention, MoE FFN, generation |
| LoRA | class | model/model_lora.py:6 | Low-rank adaptation wrapper |
| PretrainDataset | class | dataset/lm_dataset.py:37 | Raw text dataset |
| SFTDataset | class | dataset/lm_dataset.py:58 | Instruction-tuning dataset |
| DPODataset | class | dataset/lm_dataset.py | Preference pairs dataset |
| RLAIFDataset | class | dataset/lm_dataset.py | RL from AI feedback dataset |
| AgentRLDataset | class | dataset/lm_dataset.py | Agentic tool-use dataset |
| create_rollout_engine | fn | trainer/rollout_engine.py | Creates generation engine for RL training |
| setup_seed | fn | trainer/trainer_utils.py | Reproducible seeding (random, numpy, torch) |
| init_distributed_mode | fn | trainer/trainer_utils.py | DDP setup, falls back to DataParallel |
| get_lr | fn | trainer/trainer_utils.py | Cosine LR schedule with warmup |
| unwrap_model | fn | trainer/trainer_utils.py | Safe DDP/DataParallel unwrapping |

## CONVENTIONS
- **sys.path hack**: Every trainer/script does `__package__ = "trainer"` + `sys.path.append(...)` to import from root. This is the project's module resolution pattern.
- **No package manager**: No setup.py, no pyproject.toml. Direct `python script.py` execution from repo root.
- **Weight naming**: Checkpoints saved as `out/{weight_name}_{hidden_size}.pth` (e.g., `full_sft_768.pth`, `full_sft_768_moe.pth`).
- **Logger pattern**: `from trainer.trainer_utils import Logger` - prints only on main process in DDP.
- **All training scripts share identical structure**: argparse → init_model → train_epoch loop → checkpoint.

## ANTI-PATTERNS (THIS PROJECT)
- **Do NOT retrain tokenizer**: `trainer/train_tokenizer.py` header explicitly warns against it. Use provided tokenizer.
- **No test infrastructure**: Zero test files, no pytest config. Verification is manual/visual.
- **No CI/CD**: No `.github/workflows`, no Makefile. Manual execution only.
- **`warnings.filterwarnings('ignore')`**: Every training script suppresses all warnings.

## UNIQUE STYLES
- Emoji dividers in model_minimind.py: `🌏🌎🌍` separators between config sections.
- Cosine LR with custom formula: `lr*(0.1 + 0.45*(1 + math.cos(...)))` - not standard cosine decay.
- MoE parameter counting: `get_model_params()` computes active vs total params for MoE models.
- Repetition penalty in PPO/GRPO: regex-based n-gram penalty (`rep_penalty()`).

## COMMANDS
```bash
# Install deps
pip install -r requirements.txt

# Pretrain
python trainer/train_pretrain.py

# SFT
python trainer/train_full_sft.py

# DPO (RLHF)
python trainer/train_dpo.py

# PPO
python trainer/train_ppo.py

# GRPO
python trainer/train_grpo.py

# LoRA
python trainer/train_lora.py

# Distillation
python trainer/train_distillation.py

# Agent RL
python trainer/train_agent.py

# Interactive chat
python eval_llm.py

# OpenAI API server
python scripts/serve_openai_api.py

# Web demo
python scripts/web_demo.py
```

## NOTES
- `.gitignore` excludes `out/` (checkpoint dir), `website/`, `docs-minimind/`.
- `logs_69145167707.zip` and `gvim_9.2.0000_x64.exe` are stray files, not part of the project.
- Model supports both native PyTorch weights (`.pth`) and HuggingFace format (`trust_remote_code=True`).
- DDP mode requires `RANK` and `LOCAL_RANK` env vars; falls back to DataParallel automatically.
- MoE models use `_moe` suffix in checkpoint names.
