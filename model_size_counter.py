import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class RAMSizeCounter:
    def __init__(self, model, input_size=(1,1,32,32), bits=32, device='cpu'):
        self.model = model
        self.input_size = input_size
        self.bits = bits
        self.device = device
        return

    def count_parameters_size(self):
        mods = list(self.model.modules())
        sizes = []
        
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def count_output_size(self):
        input_ = Variable(torch.FloatTensor(*self.input_size))
        out_sizes = []
        with torch.no_grad():
            out = self.model(input_.to(self.device))
            out_sizes.append(np.array(out.size()))

        self.out_sizes = out_sizes
        return

    def count_parameters_ram(self):
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def count_passes_ram(self):
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.forward_backward_bits = (total_bits * 2)
        return

    def count_input_ram(self):
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def count_megabytes(self):
        self.count_parameters_size()
        self.count_output_size()
        self.count_parameters_ram()
        self.count_passes_ram()
        self.count_input_ram()
        result = self.param_bits + self.forward_backward_bits + self.input_bits

        megabytes = (result / 8) / (1024 ** 2)
        print("Model size in megabytes:", megabytes)
        return megabytes
