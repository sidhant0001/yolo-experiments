import torch

import torch
import sys
print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

def main() -> None:
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("Running on CPU.")


if __name__ == "__main__":
    main()