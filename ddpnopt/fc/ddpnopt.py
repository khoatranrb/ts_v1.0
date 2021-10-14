import torch
from torch.optim import _functional as F
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
from torch import Tensor
# import ddp

@torch.no_grad()
def update_dpp(net,grads,grads_batch,square_avgs,list_x,x_grads,lr,alpha,eps):
    x_hat = list_x[0].clone().detach()
    # x = list_x[0].clone().detach()
    params = list(net.parameters())
    for i, (module, param) in enumerate(zip(net.children(), params)):
        grad = grads[i]
        x = list_x[i//2]
        square_avg = square_avgs[i]

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        avg = square_avg.sqrt().add_(eps)
        if i % 2 == 1:
            param.addcdiv_(grad, avg, value=-lr)
            x_hat = module(x_hat)
            continue
        Q_u = grads_batch[i]*(2**((len(params)-i)//2))
        Q_uu = alpha * square_avg + (1 - alpha) * Q_u * Q_u
        Q_uu = torch.sqrt(Q_uu) + eps
        Q_x = x_grads[i//2]*(2**((len(params)-i)//2))
        Q_ux = calc_q_ux(Q_u, Q_x.unsqueeze(-1))
        small_k = calc_small_k(Q_uu, Q_u)
        big_k = calc_big_k(Q_uu, Q_ux)
        term1 = small_k.mean(dim=0)
        term2 = torch.einsum('bxy,bzt->bxt', big_k, (x_hat - x).unsqueeze(-1)).squeeze(-1).mean(dim=0).reshape(param.shape)
        # print('term1', term1.max(), term1.min(), term1.mean())
        # print('term2', term2.max(), term2.min(), term2.mean())
        h, w = term1.shape[:2]
        # print(((term1*term2)>0).sum()/(h*w),((term1*term2)<0).sum()/(h*w))
        param.add_(term1+term2, alpha=lr)
        # param.add_(term1, alpha=lr)
        x_hat = module(x_hat)

class DDPNOPT(Optimizer):
    def __init__(self, net, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        self.net = net
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(DDPNOPT, self).__init__(net.parameters(), defaults)

    def __setstate__(self, state):
        super(DDPNOPT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, x_grads, list_x, closure=None):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            grads_batch = []
            square_avgs = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                grads_batch.append(p.grad1)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                state['step'] += 1

            # list_K = grad_ddp(params_with_grad,
            #       grads_batch,
            #       square_avgs,
            #       x_grads,
            #       alpha=group['alpha'],
            #       eps=group['eps'],
            #       lr=group['lr'])
            update_dpp(self.net, grads,grads_batch,square_avgs,list_x,x_grads,group['lr'],group['alpha'],group['eps'])

def bkron(A: torch.Tensor, B: torch.Tensor):
    assert A.dim() == 3 and B.dim() == 3

    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0),
                                                    A.size(1) * B.size(1),
                                                    A.size(2) * B.size(2)
                                                    )
    return res

def calc_q_ux(q_u, q_x):
    return torch.einsum('bxy,bzt->bxt', q_u.flatten(start_dim=1).unsqueeze(-1), torch.transpose(q_x,1,2))

def calc_gain(q_ux, q_uu, q_u):
    out = q_u/q_uu
    return torch.einsum('bxy,bzt->bxt', q_ux.transpose(1,2),out.flatten(start_dim=1).unsqueeze(-1)).squeeze(-1)

def calc_v_x(q_x, gain):
    return q_x
    return q_x - gain

def calc_small_k(q_uu, q_u):
    return -q_u/q_uu
def calc_big_k(q_uu, q_ux):
    # return -q_ux
    return -torch.einsum('bx,bxt->bxt', 1/q_uu.flatten(start_dim=1), q_ux)
