import torch
from torch.utils import cpp_extension


print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print(torch._C._GLIBCXX_USE_CXX11_ABI)