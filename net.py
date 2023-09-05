import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn.utils.prune import l1_unstructured, remove, is_pruned
from torch import optim
import snn


def tospikes(x, device):
    x = (x+1.0)/2.0  # map from [-1,1] to [0,1]
    x = torch.greater_equal(x, torch.rand(
        x.shape[0], x.shape[1]).to(device)).float().to(device)
    return x


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):  # initialization
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Network(nn.Module):
    def __init__(self, args, run_name, device):
        super(Network, self).__init__()
        self.checkpoint_file = f"{run_name}"
        self.checkpoint_dir = ""
        self.args = args
        self.modules_list = nn.ModuleList()
        self.device = device

    def forward(self, input):
        device = self.device
        for index, module in enumerate(self.modules_list):
            if index == 0:
                spk = module(input)
            else:
                spk = module(spk)
        return spk

    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.state_dict(),
                   f"{self.checkpoint_dir}/{self.checkpoint_file}")

    def load(self, checkpoint_file):
        self.load_state_dict(torch.load(
            f"{self.checkpoint_dir}/{checkpoint_file}"))

    def load_current(self):
        self.load_state_dict(torch.load(
            f"{self.checkpoint_dir}/{self.checkpoint_file}"))


class Actor(Network):
    def __init__(self, input, output, args, run_name, device):
        super(Actor, self).__init__(args, run_name, device)
        self.checkpoint_dir = "models/actor"
        hidden = args.actor_shape
        if len(hidden) > 0:
            self.modules_list.append(snn.Perceptron(
                input, hidden[0], args, device))
            for i in range(0, len(hidden)-1):
                self.modules_list.append(
                    snn.Perceptron(hidden[i], hidden[i+1], args, device))
            self.modules_list.append(snn.Perceptron(
                hidden[-1], output, args, device))
        else:
            self.modules_list.append(
                snn.Perceptron(input, output, args, device, std=0.01))
        self.optimizer = optim.Adam(
            self.parameters(), lr=args.learning_rate, eps=1e-5)


class Critic(Network):
    def __init__(self, input, output, args, run_name, device):
        super(Critic, self).__init__(args, run_name, device)
        self.checkpoint_dir = "models/critic"

        hidden = args.critic_shape
        if len(hidden) > 0:
            self.modules_list.append(snn.Perceptron(
                input, hidden[0], args, device))
            for i in range(0, len(hidden)-1):
                self.modules_list.append(
                    snn.Perceptron(hidden[i], hidden[i+1], args, device))
            self.modules_list.append(
                snn.Perceptron(hidden[-1], 1, args, device))
        else:
            self.modules_list.append(snn.Perceptron(input, 1, args, device))
        self.optimizer = optim.Adam(
            self.parameters(), lr=args.learning_rate, eps=1e-5)
