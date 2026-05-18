# dataset/

## OVERVIEW
PyTorch Dataset implementations for all training stages (pretrain → SFT → DPO → RLAIF → agent RL).

## WHERE TO LOOK

| Dataset | Class | Purpose |
|---------|-------|---------|
| Raw text | `PretrainDataset` | Next-token prediction, HuggingFace JSON loader |
| Instruction | `SFTDataset` | Conversation format, system prompts, reasoning_content, tools |
| Preference | `DPODataset` | Chosen/rejected pairs, loss mask generation |
| RL from AI | `RLAIFDataset` | Thinking toggle (probability-based) |
| Tool-use | `AgentRLDataset` | Tool calls, gt (ground truth) passthrough |

| Function | Location | Role |
|----------|----------|------|
| `pre_processing_chat` | lm_dataset.py:9 | Adds system prompt (20% chance). **Preserves tool_use data untouched** |
| `post_processing_chat` | lm_dataset.py:31 | Removes empty `<think>\n\n</think>` tags (80% probability) |

## CONVENTIONS

| Pattern | Implementation |
|---------|---------------|
| Loading | `load_dataset('json', data_files=path, split='train')` |
| Return | `(input_ids, labels)` tuple, `-100` for padding |
| System prompts | 10 predefined (5 Chinese + 5 English) |
| Feature schema | `Features({conversations: [...]})` for SFTDataset |

## ANTI-PATTERNS

| Trap | Why |
|------|-----|
| Modifying `pre_processing_chat` | Breaks tool_use passthrough. Tool data passes through unchanged when `any(conv.get('tools'))` is true |
| Assuming deterministic tag removal | `post_processing_chat` uses `random.random() > 0.2` = 80% chance. Not guaranteed |
| JSON vs JSONL | All load via HuggingFace `json` loader. File extension must be `.json` for PretrainDataset, `.jsonl` for SFT/RL datasets |