from keras.datasets import mnist
import numpy as np
import torch.nn as nn
from models import Net

(train_X, train_y), (test_X, test_y) = mnist.load_data()

seq = [nn.Conv2d(1, 64, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 32, 3, 1, 1), nn.Sigmoid(),
       nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid()
       ]
net = Net(seq)

from hack_grads import *
add_hooks(net)

from torch.optim import Adam
from ddpnopt import DDPNOPT

opt = DDPNOPT(net.seq, lr=1e-3)
# opt = Adam(net.parameters(), lr=1e-3)
x = train_X[0:2]/255
x = x[:,np.newaxis,...]
# y = torch.tensor(x[:,:,:20,:20].astype(np.float32))
y = torch.tensor(x.astype(np.float32)).repeat((1,3,1,1))
x = torch.tensor(x.astype(np.float32))

mse = nn.MSELoss()

for _ in range(20):
    pred,list_x = net(x)
    loss = mse(pred,y)
    loss.backward(retain_graph=True)
    compute_grad1(net)
    x_grads = []
    for i, p in enumerate(net.parameters()):
        if i>=net.num_x: break
        x_grads.append(p.grad)
    clear_backprops(net)
    try: opt.step(x_grads,list_x)
    except: opt.step()
    # opt.step(x_grads,list_x)
    # opt.step()
    print(loss)
disable_hooks()
