import torch
import numpy as np
import torch.nn as nn


class Izi(nn.Module):
    def __init__(self,input_size,args):
        super(Izi,self).__init__()
        
        self.a = args.snn_a
        self.b = args.snn_b
        self.c = args.snn_c
        self.d = args.snn_d
        self.initial_mem_v = args.snn_c * torch.ones((1,input_size))
        self.initial_mem_r = torch.zeros((1,input_size))
        self.mem_v=self.initial_mem_v.clone()
        self.mem_r = self.initial_mem_r.clone()
        self.num_steps = args.snn_num_steps
        self.dt = 1/self.num_steps  

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        spk = torch.zeros_like(input)
        for i in range(input.shape[0]):
            self.mem_v = (4*self.mem_v**2 + 5 * self.mem_v + 1.4 - self.mem_r + input[i,:]) * self.dt
            self.mem_r = self.a * (self.b * self.mem_v - self.mem_v) * self.dt
            for j in range(self.mem_v.shape[1]):  
                if self.mem_v[0,j] >= 0.30:
                    self.mem_v[0,j] = self.c
                    self.mem_r[0,j] = self.d + self.mem_r[0,j]
                    spk[i,j] = 1
        return spk           
           

    def reset_state(self):
        self.mem_v=self.initial_mem_v.clone()
        self.mem_r = self.initial_mem_r.clone()                      
