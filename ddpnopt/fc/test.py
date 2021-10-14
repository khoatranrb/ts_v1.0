from keras.datasets import mnist
import numpy as np
import torch.nn as nn
from models import Net

(train_X, train_y), (test_X, test_y) = mnist.load_data()

seq = [nn.Linear(784,32), nn.Sigmoid(),
       nn.Linear(32,16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16, 16), nn.Sigmoid(),
       nn.Linear(16,8), nn.Sigmoid(),
       nn.Linear(8,10), nn.Softmax(1)
       ]
net = Net(seq)

from hack_grads import *
add_hooks(net)

from torch.optim import Adam
from ddpnopt import DDPNOPT

opt = DDPNOPT(net.seq, lr=1e-2)
# opt = Adam(net.parameters(), lr=1e-2)
x = train_X[0:2].reshape(2,784)/255

x = torch.tensor(x.astype(np.float32))
y = torch.tensor(np.array([[0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0]], dtype=np.float32))

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
    print(loss)
disable_hooks()
