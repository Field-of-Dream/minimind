# AGENTS.md: model/

## OVERVIEW
Core neural network architecture for MiniMind LLM. GPT-style model with MoE support, LoRA adaptation, and native generation.

## WHERE TO LOOK

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| MiniMindConfig | class | model_minimind.py:10 | Hyperparameters: hidden_size, layers, MoE, RoPE |
| MiniMindForCausalLM | class | model_minimind.py:234 | Main model: embedding, layers, lm_head, generation |
| RMSNorm | class | model_minimind.py:50 | Root mean square normalization |
| Attention | class | model_minimind.py:91 | Multi-head attention, flash-attn fallback |
| FeedForward | class | model_minimind.py:136 | Standard FFN (SwiGLU activation) |
| MOEFeedForward | class | model_minimind.py:148 | Mixture of Experts with routing |
| MiniMindBlock | class | model_minimind.py:178 | Single transformer layer |
| MiniMindModel | class | model_minimind.py:196 | Backbone without lm_head |
| precompute_freqs_cis | fn | model_minimind.py:62 | RoPE frequency precomputation with YARN scaling |
| apply_rotary_pos_emb | fn | model_minimind.py:80 | Apply rotary position embeddings |
| LoRA | class | model_lora.py:6 | Low-rank adaptation wrapper |
| apply_lora | fn | model_lora.py:21 | Inject LoRA into model layers |
| load_lora / save_lora | fn | model_lora.py:35,45 | LoRA checkpoint save/load |
| merge_lora | fn | model_lora.py:56 | Merge LoRA weights into base model |

## CONVENTIONS

- **Emoji dividers**: 🌏🌎🌍 separates config/model sections
- **Flash attention**: enabled by default (`config.flash_attn=True`)
- **Weight sharing**: `tie_word_embeddings=True` ties lm_head ↔ embed_tokens
- **MoE checkpoint suffix**: `_moe` in filenames
- **LoRA init**: Matrix A Gaussian (std=0.02), Matrix B zeros
- **RoPE + YARN**: `precompute_freqs_cis` with scaling factor for context extension

## ANTI-PATTERNS

- **No standalone test infra**: model/ has zero pytest files
- **Avoid modifying MoE routing** without understanding aux_loss calculation (router_aux_loss_coef)
- **Flash attention auto-disabled**: when `use_cache=True` or `attention_mask` is provided