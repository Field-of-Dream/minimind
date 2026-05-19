# MiniMind 多卡训练使用指南 (Kaggle T4×2)

## 概览

本指南说明如何在 Kaggle 双 T4（或 P100）GPU 上正确运行 MiniMind 训练，保证 loss 正常收敛。

## 环境要求

### Kaggle Notebook 设置

```
Accelerator → GPU T4 x2  （或 GPU P100）
```

> **P100 用户注意**：P100（Compute Capability 6.0）不支持 bfloat16。代码会自动检测并**降级到 float16**，无需加额外参数。

### 首次运行：安装依赖

在 Kaggle Notebook 第一个代码格执行：

```python
!pip install -q -r /kaggle/input/minimind/requirements.txt
```

如果报错缺少包，手动补装：

```python
!pip install -q swanlab transformers datasets accelerate
```

## 两种运行方式

### 方式 A：DDP（推荐，性能最优）

DDP（DistributedDataParallel）是 PyTorch 官方推荐的多卡方案。每张 GPU 跑独立进程，梯度通过 NCCL 正确同步。

**训练命令：**

```python
!torchrun --nproc_per_node=2 /kaggle/input/minimind/trainer/train_pretrain.py \
    --dtype float32 \
    --batch_size 16 \
    --accumulation_steps 16 \
    --grad_clip 1.0 \
    --learning_rate 3e-4 \
    --epochs 2 \
    --max_seq_len 340 \
    --data_path /kaggle/input/minimind/dataset/pretrain_t2t_mini.jsonl
```

**参数说明：**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--dtype float32` | **必选** | 禁用混合精度。T4 不支持 bfloat16，float16 在部分场景不稳定 |
| `--batch_size` | 16-32 | 每张 GPU 的 batch size |
| `--learning_rate` | 3e-4 | DDP 下有效 batch 翻倍，LR 应适当降低（原单卡 5e-4 → 3e-4） |
| `--accumulation_steps` | 8-16 | 梯度累积步数。**注意**：累积在每张卡独立做，不是跨卡累积 |
| `--grad_clip` | 1.0 | 梯度裁剪，防止梯度爆炸 |

**DDP 工作原理：**

```
GPU 0: 独立进程 → 批次[0..15] → 前向 → loss → 反向 → 梯度
GPU 1: 独立进程 → 批次[16..31] → 前向 → loss → 反向 → 梯度
                                    ↓
                             NCCL All-Reduce
                             (梯度正确平均)
                                    ↓
                             optimizer.step()
                             (参数同步更新)
```

### 方式 B：DataParallel（简单，一键运行）

不想用 `torchrun` 时，直接 `python` 运行即可。代码已自动处理 DataParallel 的 gather 问题。

**训练命令：**

```python
!python /kaggle/input/minimind/trainer/train_pretrain.py \
    --dtype float32 \
    --batch_size 16 \
    --accumulation_steps 16 \
    --grad_clip 1.0 \
    --learning_rate 3e-4 \
    --epochs 2 \
    --max_seq_len 340 \
    --data_path /kaggle/input/minimind/dataset/pretrain_t2t_mini.jsonl
```

**DataParallel 工作原理：**

```
输入 [B, L]                GPU 0: [B/2, L] → 前向 → logits [B/2, L, V]
     ↓ scatter              GPU 1: [B/2, L] → 前向 → logits [B/2, L, V]
     ↓                                          ↓ gather
    模型(DataParallel)               logits [B, L, V]
                                          ↓
                                   F.cross_entropy (在主 GPU 算)
                                          ↓
                                   loss.backward() → 梯度经 Gather
                                   反向传播到各 GPU
```

## 两种方式对比

| 特性 | DDP (`torchrun`) | DataParallel (`python`) |
|------|------------------|------------------------|
| 梯度同步 | NCCL All-Reduce | 通过 autograd Gather 传递 |
| 内存效率 | 高（独立进程） | 较低（主卡负载更高） |
| 速度 | 快 | 略慢 |
| 易用性 | 需装 torchrun | 直接运行 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 关于 loss 不收敛的排查

### 历史问题回顾

之前 loss 从 7.97 → 10.0160 的原因可能是：

1. **bfloat16 不稳定** → 用 `--dtype float32`
2. **DataParallel 梯度错误** → 已修复为 loss-after-gather
3. **学习率过大** → 多卡有效 batch 翻倍，LR 要减半
4. **梯度累积理解偏差** → 每张卡独立累积，optimizer.step() 时梯度已 all-reduce 平均

### 推荐的 loss 收敛参数

```
单卡 (原):  batch_size=32, lr=5e-4,  accumulation=8  => eff_batch=256
多卡 DDP:  batch_size=16, lr=3e-4,  accumulation=16 => eff_batch=512
多卡 DP:   batch_size=16, lr=3e-4,  accumulation=16 => eff_batch=512
```

核心原则：**effective batch size 增大 → learning rate 降低**

### 梯度验证工具

在训练脚本开头可调用新增的 `validate_gradients()` 检查梯度是否正常：

```python
from trainer.trainer_utils import validate_gradients

# 训练几个 step 后：
valid, info = validate_gradients(model)
if not valid:
    print(f"梯度异常! {info}")
```

`validate_gradients()` 会检查：
- 梯度总 norm（应为正值）
- 零梯度参数比例（超过 50% 表示梯度没流回来）
- NaN/Inf 梯度

## 训练流程（Kaggle 完整示例）

```python
# Cell 1: 安装依赖
!pip install -q -r /kaggle/input/minimind/requirements.txt

# Cell 2: DDP 训练
!torchrun --nproc_per_node=2 /kaggle/input/minimind/trainer/train_pretrain.py \
    --dtype float32 \
    --batch_size 16 \
    --accumulation_steps 16 \
    --grad_clip 1.0 \
    --learning_rate 3e-4 \
    --epochs 2 \
    --max_seq_len 340 \
    --data_path /kaggle/input/minimind/dataset/pretrain_t2t_mini.jsonl \
    --log_interval 50

# Cell 3: 测试推理
!python /kaggle/input/minimind/eval_llm.py
```

## 常见问题

**Q: `RuntimeError: Expected to have finished reduction...`**
A: DDP 下某张卡提前结束了 forward。检查是不是 `DistributedSampler` 没让各卡看到相等数量样本。

**Q: 日志里 loss 是 NaN**
A: 梯度爆炸。降低 `--learning_rate` 或增大 `--grad_clip` 到 0.5。

**Q: loss 收敛非常慢**
A: 可能 learning_rate 太低。尝试 5e-4，同时监控梯度 norm。

**Q: Kaggle 说 "RuntimeError: NCCL error"**
A: 网络通信问题。重开 Notebook 重试，或在命令前加 `export NCCL_DEBUG=INFO` 查看详情。

**Q: P100 运行报错 "bfloat16 is not supported"**
A: 已自动修复！代码会检测 P100（CC 6.0）并自动降级到 float16。如果你用的是旧版代码，手动加 `--dtype float16`。

**Q: P100 上的 float16 精度够吗？**
A: 够。P100 对 float16 有原生硬件支持（吞吐是 float32 的 2 倍），配合 `GradScaler`（已启用）可以防止梯度下溢。

**Q: 单卡能跑但多卡更慢**
A: T4/P100 的 NVLink 带宽有限，小模型可能多卡收益不大。检查 `batch_size` 是否足够大（每卡至少 16）。
