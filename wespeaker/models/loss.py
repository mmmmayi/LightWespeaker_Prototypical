import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class Angleproto(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(Angleproto, self).__init__()
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()


    def forward(self, x):

        assert x.size()[1] >= 2

        out_anchor      = x[::2,:]
        out_positive    = x[1::2,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(cos_sim_matrix, label)

        return nloss