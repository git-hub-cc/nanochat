from collections import deque
import os
import torch
import pyarrow.parquet as pq
from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=1, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    修复版 DataLoader：
    1. tokenizer_threads 默认为 1，防止 Windows 报错。
    2. 修复了数据量小时无法进入下一个 Epoch 的无限死循环 Bug。
    3. 修复了只有一个 Parquet 文件时训练集为空的 Bug。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    def document_batches():
        parquet_paths = list_parquet_files()
        
        # --- FIX 1: 防止单文件时切片为空 ---
        if len(parquet_paths) == 0:
            raise RuntimeError("No parquet files found in base_data/")
        
        if split == "train":
            # 如果只有1个文件，训练集就用这就这1个，否则留最后一个给验证集
            target_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths
        else:
            # 验证集始终取最后一个
            target_paths = parquet_paths[-1:]

        if len(target_paths) == 0:
             print(f"Warning: {split} dataset is empty! Fallback to using all files.")
             target_paths = parquet_paths

        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx 
        
        while True: # iterate infinitely (multi-epoch)
            # --- FIX 2: 这里的循环逻辑重写，确保索引重置 ---
            if pq_idx >= len(target_paths):
                pq_idx = 0 # 跑完一轮，重置回第一个文件，开始新 Epoch
                
            filepath = target_paths[pq_idx]
            try:
                pf = pq.ParquetFile(filepath)
                
                # 确定从哪个 row group 开始
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1 
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank

                # 遍历当前文件的 row groups
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size 
                
                # 当前文件读完，指向下一个
                pq_idx += 1
                
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                pq_idx += 1 # 出错跳过
                
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque() 
    
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            try:
                doc_batch, (pq_idx, rg_idx) = next(batches)
                # --- FIX 3: 强制单线程，防止 Windows Tiktoken 死锁 ---
                safe_threads = 1 if os.name == 'nt' else tokenizer_threads
                token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=safe_threads)
                for tokens in token_lists:
                    token_buffer.extend(tokens)
            except StopIteration:
                # Should not happen with infinite loop, but safety first
                break
                
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) 
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        
        # --- FIX 4: DirectML 兼容性，如果是 DML 设备，先不设 non_blocking ---
        is_dml = (str(device).startswith("privateuseone") or str(device).startswith("dml"))
        safe_non_blocking = use_cuda_optimizations and not is_dml

        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=safe_non_blocking)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=safe_non_blocking)
        
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} 
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets