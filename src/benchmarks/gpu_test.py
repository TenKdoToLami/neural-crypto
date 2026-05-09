import sys
import os
import torch
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.classifier import NeuralSentinelV1

def benchmark_batch_size(model, seq_len=100, features=8, max_batch=2048):
    print(f"{'Batch Size':<12} | {'Memory (GB)':<12} | {'Throughput (it/s)':<18}")
    print("-" * 50)
    
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    
    for batch_size in batch_sizes:
        if batch_size > max_batch: break
        
        try:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(batch_size, seq_len, features).to(device)
            
            # Warmup
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for _ in range(10):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            iterations = 50
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    for _ in range(iterations):
                        _ = model(dummy_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = (iterations * batch_size) / duration
            mem_used = torch.cuda.max_memory_allocated(device) / (1024**3)
            
            print(f"{batch_size:<12} | {mem_used:<12.2f} | {throughput:<18.2f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:<12} | OOM          | -")
                break
            else:
                raise e

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a system with RTX 4070.")
    else:
        print("Benchmarking NeuralSentinelV1 on RTX 4070...")
        model = NeuralSentinelV1()
        benchmark_batch_size(model)
