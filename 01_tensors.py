import torch

print("PyTorch Tensors")
print("=" * 40)

# Create tensors
x = torch.tensor([1, 2, 3, 4])
y = torch.randn(2, 3)

print(f"Tensor from list: {x}")
print(f"Random tensor:\n{y}")

# Check GPU
if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
