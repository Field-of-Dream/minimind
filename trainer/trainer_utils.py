"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def gpu_supports_bf16(device='cuda'):
    """检测当前 GPU 是否支持 bfloat16。（P100/Pascal 不支持，V100/T4+ 支持）"""
    if not torch.cuda.is_available():
        return False
    try:
        cc = torch.cuda.get_device_capability(device)  # (major, minor)
        # bfloat16 原生支持需要 CC 8.0+ (Ampere)；CC 7.0+ (Volta/Turing) 可通过软件转换
        supports = cc >= (7, 0)
        if not supports:
            Logger(f"  → GPU (CC {cc[0]}.{cc[1]}) 不支持 bfloat16")
        return supports
    except Exception:
        return False


def init_distributed_mode():
    """初始化分布式模式。
    
    返回: local_rank (int)
      - DDP模式 (torchrun): 返回实际 local_rank
      - 非DDP模式: 返回 0（由 DataParallel 处理多卡）
    """
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if rank == -1:
        n_gpu = torch.cuda.device_count()
        Logger(f"  → 非 DDP 模式 (RANK 未设置)")
        if n_gpu > 1:
            Logger(f"  → 使用 DataParallel 多卡训练，GPU 数量: {n_gpu}")
            Logger(f"  → GPU 列表: {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")
        return 0
    
    # DDP mode
    Logger(f"  → DDP 模式: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    Logger(f"  → 使用 GPU: {torch.cuda.get_device_name(local_rank)}")
    return local_rank


def get_multi_gpu_info():
    """返回当前 GPU 拓扑和分布式状态信息（用于调试）"""
    info = {
        "available_gpus": torch.cuda.device_count(),
        "cuda_available": torch.cuda.is_available(),
        "ddp_active": dist.is_initialized(),
    }
    if dist.is_initialized():
        info["world_size"] = dist.get_world_size()
        info["rank"] = dist.get_rank()
    info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return info


def validate_gradients(model, threshold=1e-8):
    """验证模型梯度是否正常流动（返回 True/False + 诊断信息）
    
    用于 DataParallel 多卡训练时检查梯度是否正确同步。
    如果某个参数的 grad_norm 为0或 NaN，说明梯度没有正确流回。
    """
    import math
    total_params = 0
    zero_grad = 0
    nan_grad = 0
    total_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        total_params += 1
        norm = param.grad.norm().item()
        total_norm += norm ** 2
        
        if norm < threshold:
            zero_grad += 1
        if math.isnan(norm) or math.isinf(norm):
            nan_grad += 1
    
    total_norm = total_norm ** 0.5
    valid = (zero_grad < total_params * 0.5) and (nan_grad == 0) and (total_norm > 0)
    
    return valid, {
        "total_norm": round(total_norm, 6),
        "zero_grad_params": zero_grad,
        "total_params": total_params,
        "nan_grad_params": nan_grad,
        "healthy": valid
    }


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 统一管理并行包装类型
PARALLEL_WRAPPERS = (DistributedDataParallel, nn.DataParallel)


def unwrap_model(model):
    """安全解包 DDP 或 DataParallel 包装，返回原始模型。"""
    while hasattr(model, 'module') and isinstance(model, PARALLEL_WRAPPERS):
        model = model.module
    # 兼容 torch.compile
    model = getattr(model, '_orig_mod', model)
    return model

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        raw_model = unwrap_model(model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = unwrap_model(value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path=None, save_dir=None, device='cuda'):
    # Resolve paths relative to the repository root (trainer_utils.py is in trainer/)
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if tokenizer_path is None:
        tokenizer_path = os.path.join(_repo_root, 'model')
    if save_dir is None:
        save_dir = os.path.join(_repo_root, 'out')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = os.path.join(save_dir, f'{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth')
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)