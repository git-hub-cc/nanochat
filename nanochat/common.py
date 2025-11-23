"""
Common utilities for nanochat.
MODIFIED for Windows AMD GPU (DirectML) support.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# --- 新增：尝试导入 DirectML ---
try:
    import torch_directml
except ImportError:
    torch_directml = None

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"
    if os.path.exists(file_path): return file_path
    with FileLock(lock_path):
        if os.path.exists(file_path): return file_path
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")
        if postprocess_fn is not None:
            postprocess_fn(file_path)
    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # --- 修改点：增加 dml 检测 ---
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    elif torch_directml is not None and torch_directml.is_available():
        device_type = "dml" # DirectML for AMD on Windows
    else:
        device_type = "cpu"

    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"):
    """Basic initialization."""

    if device_type == "cuda" and not torch.cuda.is_available():
        print0("WARNING: CUDA not available, falling back to CPU.")
        device_type = "cpu"

    if device_type == "mps" and not torch.backends.mps.is_available():
        print0("WARNING: MPS not available, falling back to CPU.")
        device_type = "cpu"

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision("high")

    # Distributed setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        backend = "nccl" if os.name != 'nt' else "gloo"
        dist.init_process_group(backend=backend, device_id=device)
        dist.barrier()
    elif device_type == "dml":
        # --- 修改点：初始化 DirectML 设备 ---
        if torch_directml is None:
            raise ImportError("torch-directml not installed. Please run `uv pip install torch-directml`")
        device = torch_directml.device(torch_directml.default_device())
        print0(f"Using DirectML device: {device}")
        # DirectML 通常不支持 DDP，这里强制单卡
        ddp = False
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    def __init__(self): pass
    def log(self, *args, **kwargs): pass
    def finish(self): pass