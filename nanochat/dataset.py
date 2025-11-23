"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- converting local MD files to parquet format for training

MODIFIED for local AMD GPU training on custom Markdown corpus.
"""

import os
import glob
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir, print0

# -----------------------------------------------------------------------------
# 配置本地数据源路径
# 请确保该路径下有你的 .md 文件
LOCAL_MD_SOURCE_DIR = r"C:\Users\wyswydx\IdeaProjects\Blog\blog\md"

# 处理后的 parquet 文件存放位置 (通常在 ~/.cache/nanochat/base_data)
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    # 如果没有 parquet 文件，尝试先从本地 md 转换
    data_dir = DATA_DIR if data_dir is None else data_dir
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])

    if not files:
        print0(f"No parquet files found in {data_dir}. Converting local MD files...")
        prepare_local_data()
        files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        ])

    parquet_paths = [os.path.join(data_dir, f) for f in files]
    return parquet_paths

def prepare_local_data():
    """
    扫描 LOCAL_MD_SOURCE_DIR 下的所有 .md 文件，
    读取内容，并将其保存为 parquet 格式以便 dataloader 读取。
    """
    if not os.path.exists(LOCAL_MD_SOURCE_DIR):
        print0(f"Error: Local source directory not found: {LOCAL_MD_SOURCE_DIR}")
        return

    print0(f"Scanning for .md files in {LOCAL_MD_SOURCE_DIR}...")
    # 递归查找所有 .md 文件
    md_files = glob.glob(os.path.join(LOCAL_MD_SOURCE_DIR, "**/*.md"), recursive=True)
    print0(f"Found {len(md_files)} markdown files.")

    if not md_files:
        return

    docs = []
    total_chars = 0

    for fpath in md_files:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if text:
                    docs.append(text)
                    total_chars += len(text)
        except Exception as e:
            print0(f"Failed to read {fpath}: {e}")

    print0(f"Total characters loaded: {total_chars:,}")

    # 将数据保存为一个或多个 parquet 文件
    # 为了简单起见，这里如果不超过几百兆，保存为一个文件即可
    # 模拟分片：如果数据量大，可以切分，这里简单保存为 shard_00000.parquet
    # 另外保留一部分作为验证集 (shard_00001.parquet)

    split_idx = int(len(docs) * 0.95) # 95% 训练，5% 验证
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]

    if train_docs:
        _save_to_parquet(train_docs, os.path.join(DATA_DIR, "shard_00000.parquet"))
        print0(f"Saved training data to shard_00000.parquet ({len(train_docs)} docs)")

    if val_docs:
        # 注意：dataloader 逻辑是取最后一个文件作为验证集
        # 所以我们需要确保至少有2个文件，或者文件名排序正确
        _save_to_parquet(val_docs, os.path.join(DATA_DIR, "shard_00001.parquet"))
        print0(f"Saved validation data to shard_00001.parquet ({len(val_docs)} docs)")

def _save_to_parquet(doc_list, output_path):
    """辅助函数：保存文本列表为 parquet"""
    # 使用 pandas/pyarrow 保存，这是最稳健的方法
    df = pd.DataFrame({"text": doc_list})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()

    if len(parquet_paths) == 0:
        raise RuntimeError("No data found! Please check your MD directory.")

    # 简单的划分逻辑：最后一个文件作为验证集，其余作为训练集
    # 如果只有一个文件，不仅无法验证，dataloader 也会报错，所以 prepare_local_data 保证了至少切分出验证集
    if len(parquet_paths) > 1:
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    else:
        # 只有1个文件时的 fallback（虽然上面逻辑尽量避免了）
        if split == "val":
            print0("Warning: Only 1 shard found, using it for validation too (data leakage!)")

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

if __name__ == "__main__":
    # 手动运行此脚本可触发数据转换
    prepare_local_data()