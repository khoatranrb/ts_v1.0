import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, seq):
        super(Net, self).__init__()
        self.num_x = len(seq)//2
        self.seq = nn.Sequential(*seq)
    def forward(self, x):
        list_x = [x]
        for i, module in enumerate(self.seq.children()):
            if i % 2 == 0:
                setattr(self, 'x%s' % (i//2), nn.Parameter(x))
                out1 = module(x)
                exec('out2 = module(self.x%s)'% (i//2),{'module':module, 'self':self}, globals())
                x = (out1+out2)/2
                # x = out1
            else:
                x = module(x)
                list_x.append(x)
        return x, list_x
