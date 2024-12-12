#script to ensure cuda is working

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)  # Prints the CUDA version
print(torch.backends.cudnn.version())  # Prints cuDNN version
