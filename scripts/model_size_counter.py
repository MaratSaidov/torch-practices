import numpy as np

import torch

from torch import nn
from torch.autograd import Variable


class RAMSizeCounter:
    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32, device="cpu"):
        self.model = model
        self.input_size = input_size
        self.bits = bits
        self.device = device

    def count_parameters_size(self):
        sizes = []
        for m in self.model.modules():
            for p in m.parameters():
                sizes.append(np.array(p.size()))
        self.param_sizes = sizes

    def count_output_size(self):
        input_ = Variable(torch.FloatTensor(*self.input_size))
        out_sizes = []
        with torch.no_grad():
            out = self.model(input_.to(self.device))
            out_sizes.append(np.array(out.size()))
        self.out_sizes = out_sizes

    def count_parameters_ram(self):
        bits = 0
        for s in self.param_sizes:
            bits += np.prod(np.array(s)) * self.bits
        self.param_bits = bits

    def count_passes_ram(self):
        bits = 0
        for s in self.out_sizes:
            bits += np.prod(np.array(s)) * self.bits
        self.forward_backward_bits = 2 * bits

    def count_input_ram(self):
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits

    def count_megabytes(self):
        self.count_parameters_size()
        self.count_output_size()
        self.count_parameters_ram()
        self.count_passes_ram()
        self.count_input_ram()
        result = self.param_bits + self.forward_backward_bits + self.input_bits

        megabytes = (result / 8) / (1024 ** 2)
        print(f"Model size is {megabytes} MB")
        return megabytes
