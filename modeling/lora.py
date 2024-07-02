import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim,rank,device, alpha=0.01):
        super(LoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank)* std_dev)
        self.B = nn.Parameter(torch.zeros(rank,out_dim))
        self.A = nn.Parameter(self.A.to(device))
        self.B = nn.Parameter(self.B.to(device))
        # print("A created, size:", self.A.size())
        # print("B created, size:", self.B.size())
        self.alpha = alpha
        # print("LoRALayer initialized, parameters:", list(self.parameters()))
    
    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
    
    def print_parameters(self):
        print("show LoRALayer parameters:")
        for name, param in self.named_parameters():
            print(name, param.size())

class LinearWithLoRA(nn.Module):
    def __init__(self, linear,rank,device, alpha=0.01):
        super(LinearWithLoRA, self).__init__()
        self.linear = linear

        lora = LoRALayer(linear.in_features, linear.out_features,rank,device, alpha)
        # lora.print_parameters()
        
        self.add_module('lora', lora)
        
        for param in self.linear.parameters():
            param.requires_grad = False
        for param in self.lora.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
    def print_parameters(self):
        print("show LinearWithLoRA parameters:")
        for name, param in self.named_parameters():
            print(name, param.size())
