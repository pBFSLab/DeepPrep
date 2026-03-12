import warnings
from typing import List, Optional

try:
    import pynvml
except ImportError:
    raise SystemExit("pynvml not found. Run:  pip install nvidia-ml-py")


def list_gpu_memory(free_memory: bool = True) -> List[int]:
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        mem_list = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            if free_memory:
                # 剩余显存 = 总显存 - 已用显存
                mem_list.append(mem_info.free // (1024 * 1024))
            else:
                # 总显存
                mem_list.append(mem_info.total // (1024 * 1024))

        return mem_list
    except Exception as e:
        warnings.warn(f"Failed to query GPU memory: {e}")
        return []


def pick_gpu_index(target_idx: Optional[int] = None, threshold_mib: int = 19 * 1024, free_memory: bool = True) -> Optional[int]:
    mem_list = list_gpu_memory(free_memory=free_memory)
    if not mem_list:
        return None

    if target_idx is None:
        best_idx, best_mem = max(enumerate(mem_list), key=lambda x: x[1])
        return best_idx if best_mem >= threshold_mib else None
    else:
        if 0 <= target_idx < len(mem_list) and mem_list[target_idx] >= threshold_mib:
            return target_idx
        return None


def auto_device(device_hint: str = "cuda", threshold_mib: int = 19 * 1024, free_memory: bool = True) -> str:
    device_hint = device_hint.strip().lower()

    if device_hint == "cpu":
        return "cpu"

    if device_hint.isdigit():
        idx = int(device_hint)
        chosen = pick_gpu_index(target_idx=idx, threshold_mib=threshold_mib, free_memory=free_memory)
        return f"{chosen}" if chosen is not None else "cpu"

    if device_hint in ("cuda", "auto"):
        chosen = pick_gpu_index(target_idx=None, threshold_mib=threshold_mib, free_memory=free_memory)
        return f"{chosen}" if chosen is not None else "cpu"

    if device_hint.startswith("cuda:"):
        try:
            idx = int(device_hint.split(":")[1])
        except ValueError:
            warnings.warn(f"Invalid cuda index in '{device_hint}', fallback to cpu")
            return "cpu"
        chosen = pick_gpu_index(target_idx=idx, threshold_mib=threshold_mib, free_memory=free_memory)
        return f"{chosen}" if chosen is not None else "cpu"

    warnings.warn(f"Unrecognized device hint '{device_hint}', fallback to cpu")
    return "cpu"


if __name__ == "__main__":
    # device = auto_device('0')
    # device = auto_device('cpu')
    device = auto_device('auto')
    print("Selected device:", device)
