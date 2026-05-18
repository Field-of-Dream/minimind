# trainer/ KNOWLEDGE BASE

## OVERVIEW
Training scripts for all pipeline stages: pretrain, SFT, DPO, PPO, GRPO, LoRA, distillation, agent RL.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Shared utils | `trainer_utils.py` | setup_seed, get_lr, Logger, init_distributed_mode, unwrap_model |
| Generation engine | `rollout_engine.py` | create_rollout_engine for RL loops |
| Repetition penalty | trainer_utils.py | regex-based n-gram penalty for PPO/GRPO |

## CONVENTIONS
- Module resolution: `__package__ = "trainer"` + `sys.path.append(...)` in every script
- Warnings suppressed: `warnings.filterwarnings('ignore')` in all trainers
- Logger: prints only on main process in DDP
- Checkpoint naming: `out/{weight_name}_{hidden_size}.pth`
- DDP fallback: requires RANK/LOCAL_RANK env vars, auto-falls back to DataParallel
- Custom cosine LR: `lr*(0.1 + 0.45*(1 + math.cos(...)))`

## ANTI-PATTERNS
- **Do NOT retrain tokenizer**: train_tokenizer.py header explicitly warns. Use provided tokenizer.
- **No test infrastructure**: verification is manual/visual. See root AGENTS.md.
- **No CI/CD**: manual execution only. See root AGENTS.md.