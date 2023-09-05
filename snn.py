import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def conditional_probability_A_given_not_B(A, B):
    # Calculate P(B|A) using Bayes' theorem
    P_A_given_not_B = torch.clamp(torch.max(torch.zeros_like(A), torch.add(
        input=B, other=A, alpha=-1)) + torch.min(A, 1-B)/(2*(1-B)+torch.finfo(torch.float32).eps), min=0., max=1.)

    return P_A_given_not_B


def layer_init(input_size, output_size, std=np.sqrt(2), has_bias=False, bias_const=0.0):  # initialization
    weight = torch.nn.Parameter(torch.empty(output_size, input_size))
    nn.init.orthogonal_(weight, std)
    if has_bias:
        bias = torch.nn.Parameter(torch.empty(output_size))
        nn.init.constant_(bias, bias_const)
    else:
        bias = None
    return weight, bias


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size, args, device, std=np.sqrt(2)):
        super(Perceptron, self).__init__()
        self.device = device
        self.weight, self.bias = layer_init(
            input_size, output_size, std, has_bias=args.has_bias)
        self.args = args
        self.container = stdp_container(self, args)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        spk = self.container(input, self.weight, self.bias)
        return spk


def stdp_container(neuron_layer, args):
    class forward_stdp_func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            a, b, c, d = args.snn_a, args.snn_b, args.snn_c, args.snn_d
            num_steps = args.snn_num_steps
            dt = 1/num_steps
            output = F.linear(input, weight, bias)  # TEST
            output = F.sigmoid(output)           # TEST

            if args.stdp == True:

                stdp_w_plus = conditional_probability_A_given_not_B(input.unsqueeze(1).repeat(
                    1, output.shape[1], 1), output.unsqueeze(2).repeat(1, 1, input.shape[1]))
                stdp_w_minus = conditional_probability_A_given_not_B(output.unsqueeze(2).repeat(
                    1, 1, input.shape[1]), input.unsqueeze(1).repeat(1, output.shape[1], 1))

                stdp_weight = torch.add(
                    input=stdp_w_plus*(-stdp_w_plus).exp(), other=stdp_w_minus*(-stdp_w_minus).exp(), alpha=-1)
                if args.has_bias:
                    stdp_b_plus = torch.clamp(torch.max(torch.zeros(input.shape[0], output.shape[1]).to(neuron_layer.device), torch.add(
                        input=output, other=.5*torch.ones_like(output), alpha=-1)) + torch.min(.5*torch.ones_like(output), 1-output)/(2*(1-output)+torch.finfo(torch.float32).eps), min=0., max=1.).to(neuron_layer.device)

                    stdp_b_minus = torch.clamp(torch.max(torch.zeros(input.shape[0], output.shape[1]).to(neuron_layer.device), torch.add(
                        input=.5*torch.ones_like(output), other=output, alpha=-1)) + torch.min(output, 1-.5*torch.ones_like(output))/(2*(1-.5*torch.ones_like(output))+torch.finfo(torch.float32).eps), min=0., max=1.).to(neuron_layer.device)

                    stdp_bias = torch.add(
                        input=stdp_b_plus*(-stdp_b_plus).exp(), other=stdp_b_minus*(-stdp_b_minus).exp(), alpha=-1)
                else:
                    stdp_bias = torch.empty(0)
                ctx.save_for_backward(
                    weight, stdp_weight, bias, stdp_bias)
            else:
                ctx.save_for_backward(
                    weight, bias, output, input)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_weight = None
            if args.stdp == True:
                grad_temp = grad_output
                weight, stdp_weight, bias, stdp_bias = ctx.saved_tensors
                if ctx.needs_input_grad[0]:
                    grad_input = grad_temp.mm(
                        torch.abs(weight))/torch.abs(weight).sum(0)
                if ctx.needs_input_grad[0]:
                    grad_weight = -\
                        stdp_weight.mul(grad_temp.unsqueeze(2)).sum(0)
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = \
                        stdp_bias.mul(grad_temp).sum(0)
                else:
                    grad_bias = None
            else:
                weight, bias, output, input = ctx.saved_tensors
                grad_temp = grad_output.clone()
                grad_temp = grad_temp * output * \
                    (torch.ones_like(output)-output)
                if ctx.needs_input_grad[0]:
                    grad_input = grad_temp.mm(weight)
                if ctx.needs_input_grad[1]:
                    grad_weight = grad_temp.t().mm(input)
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = grad_output.sum(0)
                else:
                    grad_bias = None
            return grad_input, grad_weight, grad_bias
    return forward_stdp_func.apply
