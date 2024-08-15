import torch
from torch import nn

class LIFCell(nn.Module):
    def __init__(self, size, tau_m, dt):
        super().__init__()
        self.v = torch.randn(size)
        self.v_th = torch.randn(size)
        self.v_rest = torch.randn(size)
        self.v_reset = torch.randn(size)
        self.R_m = torch.randn(size)
        self.tau_m = tau_m
        self.dt = dt
    def forward(self, input):
        self.v = self.v + (self.dt/self.tau_m) * (self.v_rest - self.v + self.R_m * input)
        torch.where((self.v >= self.v_th), self.v_reset, self.v)
        return self.v