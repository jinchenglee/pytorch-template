import torch
import torch.nn.functional as F


def nll_loss(output, target):
    # Convert to datatype that avoid runtime error:
    #   Expected object of type torch.cuda.LongTensor but found type torch.cuda.FloatTensor for argument #2 'target'
    target_as_LongTensor = target.type(torch.cuda.LongTensor)
    return F.nll_loss(output, target_as_LongTensor)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

