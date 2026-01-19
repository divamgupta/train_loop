
import subprocess

def get_gpu_memory_info(threshold=0.9):
    """
    Get GPU IDs with memory usage information.
    
    Args:
        threshold: Free memory threshold (default 0.9 for 90%)
    
    Returns:
        tuple: (free_gpus, used_gpus)
            - free_gpus: dict of {gpu_id: memory_usage_percent} for GPUs with >threshold free memory
            - used_gpus: dict of {gpu_id: memory_usage_percent} for other GPUs
            Both sorted by free memory (ascending usage)
    
    Raises:
        FileNotFoundError: If nvidia-smi is not found
        subprocess.CalledProcessError: If nvidia-smi fails
        ValueError: If GPU data cannot be parsed
    """
    # Query nvidia-smi for GPU memory info
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
         '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=True
    )
    
    gpu_info = []
    for line in result.stdout.strip().split('\n'):
        if line:
            # Parse: index, memory.used, memory.total
            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 3:
                raise ValueError(f"Expected 3 values, got {len(parts)}: {line}")
            
            gpu_id = int(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            
            if mem_total <= 0:
                raise ValueError(f"Invalid total memory for GPU {gpu_id}: {mem_total}")
            
            # Calculate usage percentage
            usage_percent = mem_used / mem_total
            gpu_info.append((gpu_id, usage_percent))
    
    # Sort by usage (ascending = most free first)
    gpu_info.sort(key=lambda x: x[1])
    
    # Split into free and used based on threshold
    free_gpus = {gpu_id: usage for gpu_id, usage in gpu_info 
                 if usage < (1 - threshold)}
    used_gpus = {gpu_id: usage for gpu_id, usage in gpu_info 
                 if usage >= (1 - threshold)}
    
    return free_gpus, used_gpus


def get_free_gpus_ids(num, threshold=0.9):
    """
    Get list of GPU IDs with free memory above the threshold.
    
    Args:
        threshold: Free memory threshold (default 0.9 for 90%)
    
    Returns:
        list: GPU IDs with >threshold free memory
    """
    free_gpus, _ = get_gpu_memory_info(threshold)
    free_gpus = list(free_gpus.keys())

    if len(free_gpus) < num:
        raise RuntimeError(f"Requested {num} free GPUs, but only {len(free_gpus)} available.")
    
    return sorted(free_gpus[:num])