import torch
import torch.nn as nn
x = torch.ones([2,3,3])
soft = nn.Softmax2d()

factor = torch.arange(1,19).to(torch.float32).view([2,3,3])
print(f"factor = \n{factor}")
factor = soft(factor)
print(f"factor = \n{factor}")
y = x * factor 
# print(f"x = \n{x}")

# print(f"y = \n{y}")
