import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

# Test code
batch_size = 1

def get_inputs():
    # return [torch.rand(batch_size, 1, 10240, 10240)]
    return [torch.rand(batch_size, 1, 4096, 4096)]
    # return [torch.rand(batch_size, 4, 5120, 5120)]

def get_init_inputs():
    return []
