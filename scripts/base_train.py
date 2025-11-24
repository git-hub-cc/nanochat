"""
Train model. Run as:
python -m scripts.base_train

MODIFIED for Windows AMD GPU (DirectML) + 8GB VRAM.
"""

import os
import sys
import time
from contextlib import nullcontext
import copy

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # 留空以自动检测 (会检测 dml/cuda/cpu)

# --- 显存适配配置 (Target: 8GB VRAM) ---
# 降低模型深度，标准 GPT-2 Small 是 12 层，这里设为 10 层以节省显存
depth = 10
# 上下文长度，1024 是一个比较安全的平衡点
max_seq_len = 512

# Training horizon
# --- 修改点：将迭代次数从 1000 改为 20 以进行快速验证 ---
num_iterations = 20
target_flops = -1.0
target_param_data_ratio = 20

# Optimization
# --- 显存适配配置 ---
# 批次大小设为 2，防止 OOM。如果显存还有富余，可以尝试改回 4
device_batch_size = 1
# 累积梯度的总批次大小，保持较小以加快 update 频率
total_batch_size = 65536

embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0
resume_from_step = -1

# Evaluation
# --- 修改点：调整评估间隔以适应 20 次迭代 ---
eval_every = 10
eval_tokens = 20*1024*4
core_metric_every = 20 # 在最后一步进行评估
core_metric_max_per_task = 200
sample_every = 10
save_every = 20 # 在最后一步进行保存
# -----------------------------------------------

# Output
model_tag = ""

# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# --- 关键修改：适配 DirectML 的混合精度上下文 ---
if device_type == "cuda":
    # NVIDIA 卡使用 bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    synchronize = torch.cuda.synchronize
    get_max_memory = torch.cuda.max_memory_allocated
elif device_type == "dml":
    # AMD 卡 (DirectML) 在 Windows 上使用 privateuseone + float16
    # --- FIX: 禁用 Autocast ---
    # torch-directml 的 Autocast 实现缺失大量算子（如 slice, copy_），导致 crash。
    # 我们改为禁用自动混合精度，转而手动将模型设为半精度（或者全精度）。
    print0("Warning: DirectML autocast support is incomplete. Disabling AMP (using FP32/FP16 manually).")
    autocast_ctx = nullcontext()

    # DirectML 没有显式的同步和显存查询 API，设为占位符
    synchronize = lambda: None
    get_max_memory = lambda: 0
else:
    # CPU 回退
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    synchronize = lambda: None
    get_max_memory = lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
# --- Fix: DirectML map_location bug prevention ---
# Passing device=device here might crash if device is 'privateuseone'
# But tokenizer.py has been patched to ignore the device arg in torch.load
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model Configuration
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Batch size calculation
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size

# 自动修正 total_batch_size 防止除零错误
if total_batch_size < world_tokens_per_fwdbwd:
    total_batch_size = world_tokens_per_fwdbwd
    print0(f"Adjusted total_batch_size to {total_batch_size} to match device capacity")

if total_batch_size % world_tokens_per_fwdbwd != 0:
    grad_accum_steps = max(1, total_batch_size // world_tokens_per_fwdbwd)
else:
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# --- FIX: Manual Mixed Precision for DML ---
# Since we disabled autocast, we should try to use half precision to save memory on 8GB cards.
# However, pure FP16 training can be unstable.
# If you encounter NaN losses, comment out the following if block to revert to FP32.
# if device_type == "dml":
#     print0("Converting model to float16 to save memory (Manual AMP)...")
#     # 注意：Embedding 层在 gpt.py 中对于非 CUDA 设备保持了 fp32，这里 model.half() 会尝试全部转 fp16
#     # 这对于 DirectML 通常是支持的。
#     model = model.half()

# Resume logic
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    # Ensure model stays in half precision if resumed
    if device_type == "dml":
        model = model.half()
    del model_data

orig_model = model

# --- 关键修改：Windows/DirectML 禁用 torch.compile ---
if os.name == 'nt' or device_type == 'dml':
    print0("Windows/DirectML detected: Disabling torch.compile for stability.")
    model = orig_model
else:
    print0("Compiling model with torch.compile...")
    try:
        model = torch.compile(model, dynamic=False)
    except Exception as e:
        print0(f"torch.compile failed: {e}. Fallback to eager mode.")
        model = orig_model

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Training horizon calculation
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    # Fallback default if calculation fails
    num_iterations = 1000
    print0(f"Fallback: Setting iterations to {num_iterations}")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")

# -----------------------------------------------------------------------------
# Initialize Optimizer
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# -----------------------------------------------------------------------------
# DataLoaders
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Schedulers
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Validation
    if last_step or step % eval_every == 0:
        model.eval()
        try:
            val_loader = build_val_loader()
            eval_steps = max(1, eval_tokens // (device_batch_size * max_seq_len * ddp_world_size))
            # Note: evaluate_bpb might need updates if using manual FP16, but autocast_ctx is null now
            with autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log({
                "step": step,
                "val/bpb": val_bpb,
            })
        except Exception as e:
            print0(f"Validation skipped (data insufficient or error): {e}")
        model.train()

    # Sampling
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()

        # --- FIX: Temporarily switch to float32 for stable sampling on DirectML ---
        is_half_precision = False
        if device_type == "dml" and orig_model.lm_head.weight.dtype == torch.float16:
            is_half_precision = True
            orig_model.float() # Switch to float32 for stable sampling

        prompts = [
            "The capital of China is",
            "AI is",
            "My favorite food is",
        ]
        engine = Engine(orig_model, tokenizer)
        print0("\n--- Sampling ---")
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            # autocast_ctx is nullcontext() on DML, this has no effect, which is fine
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0.8)
            print0(f"> {prompt} -> {tokenizer.decode(sample[0])}")
        print0("----------------\n")

        # --- FIX: Revert model back to original precision for training ---
        if is_half_precision:
            orig_model.half() # Switch back to float16 for training efficiency

        model.train()

    # Checkpoint
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        # Save model in FP32 to ensure compatibility
        model_to_save = orig_model
        if device_type == "dml":
            model_to_save = copy.deepcopy(orig_model).float()

        save_checkpoint(
            checkpoint_dir,
            step,
            model_to_save.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb if 'val_bpb' in locals() else 0.0,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )
        if device_type == "dml":
            del model_to_save # free memory

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    if grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip).item()
    else:
        grad_norm = 0.0

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt) if dt > 0 else 0

    if step > 10:
        total_training_time += dt

    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | grad norm: {grad_norm:.4f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | total time: {total_training_time/60:.2f}m")

    if step % 20 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        })

    step += 1

# End
wandb_run.finish()
compute_cleanup()