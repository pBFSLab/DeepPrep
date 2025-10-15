import warnings
from typing import List, Optional

try:
    import pynvml
except ImportError:
    raise SystemExit("pynvml not found. Run:  pip install nvidia-ml-py")


def list_gpu_memory() -> List[int]:
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        mem_list = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_list.append(mem_info.total // (1024 * 1024))
        return mem_list
    except Exception as e:
        warnings.warn(f"Failed to query GPU memory: {e}")
        return []


def pick_gpu_index(target_idx: Optional[int] = None, threshold_mib: int = 19 * 1024) -> Optional[int]:
    mem_list = list_gpu_memory()
    if not mem_list:
        return None
    if target_idx is None:
        best_idx, best_mem = max(enumerate(mem_list), key=lambda x: x[1])
        return best_idx if best_mem >= threshold_mib else None
    else:
        if 0 <= target_idx < len(mem_list) and mem_list[target_idx] >= threshold_mib:
            return target_idx
        return None


def auto_device(device_hint: str = "cuda", threshold_mib: int = 19 * 1024) -> str:
    """
    Supported notations:
      'cpu'           use CPU directly
      'cuda'          auto-select GPU
      '0' / '1' / …   validate the corresponding device ID; if valid use cuda:0 / cuda:1, otherwise fall back to CPU
    """
    device_hint = device_hint.strip().lower()
    if device_hint == "cpu":
        return "cpu"

    if device_hint.isdigit():
        idx = int(device_hint)
        chosen = pick_gpu_index(target_idx=idx, threshold_mib=threshold_mib)
        return f"{chosen}" if chosen is not None else "cpu"

    if device_hint == "cuda" or device_hint == "auto":
        chosen = pick_gpu_index(target_idx=None, threshold_mib=threshold_mib)
        return f"{chosen}" if chosen is not None else "cpu"

    if device_hint.startswith("cuda:"):
        try:
            idx = int(device_hint.split(":")[1])
        except ValueError:
            warnings.warn(f"Invalid cuda index in '{device_hint}', fallback to cpu")
            return "cpu"
        chosen = pick_gpu_index(target_idx=idx, threshold_mib=threshold_mib)
        return f"{chosen}" if chosen is not None else "cpu"

    warnings.warn(f"Unrecognized device hint '{device_hint}', fallback to cpu")
    return "cpu"


if __name__ == "__main__":
    # device = auto_device('0')
    # device = auto_device('cpu')
    device = auto_device('auto')
    print("Selected device:", device)
