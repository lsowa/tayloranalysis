import torch
import pickle
import time

from torch import nn
from tayloranalysis import TaylorAnalysis

# load data

data = pickle.load(open("data/data.pickle", "rb"), encoding="latin-1")
x_train = torch.tensor(data['x_train'], dtype=torch.float)
y_train = torch.tensor(data['y_train'], dtype=torch.float)

# initialize nomral pytorch model

class Mlp(nn.Module):
    
    def __init__(self, input_neurons, hidden_neurons, 
                output_neurons, hiddenlayers):
        
        nn.Module.__init__(self)

        # mlp layers
        self.mlplayers = nn.ModuleList([nn.Linear(input_neurons,
                hidden_neurons)])
        self.mlplayers.extend([nn.Linear(hidden_neurons, 
                                        hidden_neurons) for i in range(hiddenlayers + 1)])
        self.mlplayers.append(nn.Linear(hidden_neurons, output_neurons))

    def forward(self, x):
        # input shape: (batch, features)
        for mlplayer in self.mlplayers[:-1]:
            x = mlplayer(x)
            x = torch.tanh(x) 
        
        # new x: (batch, 1)
        x = self.mlplayers[-1](x)
        x = x.squeeze(-1)  # new x: (batch)
        x = torch.sigmoid(x)
        return x

model = Mlp(2, 100, 1, 2)

# wrap model

model = TaylorAnalysis(model)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.BCELoss()

device=torch.device(3)
x_train=x_train.to(device)
y_train=y_train.to(device)
model.to(device)

x_train.requires_grad = True
start = time.time()
for epoch in range(200):
    optim.zero_grad()
    pred = model(x_train)
    loss = crit(pred, y_train)
    loss.backward()
    optim.step()
    print('Epoch {}: Loss: {:.3f}'.format(epoch+1, loss))
    
    # save current taylorcoefficients
    
    #model.tc_checkpoint(x_train, names=['x1', 'x2'], order=3)
end = time.time()
print('Time needed:', round(end - start, 2))

# load test data

x_test = torch.tensor(data['x_test'], dtype=torch.float).to(device)
y_test = torch.tensor(data['y_test'], dtype=torch.float).to(device)

# plot taylorcoefficients after training

dummy = torch.tensor([0, 0], dtype=torch.float).unsqueeze(0).to(device)
model.ood_test(x_data=x_test, y_data=y_test, x=dummy, names=['x1', 'x2'], order=3)

# plot saved checkpoints

#model.plt_checkpoints()

dat = model._second_order(x_data=x_test[y_test==0], ind_i=0, raw=True)[:,1]
dat.max()
import matplotlib.pyplot as plt
_ = plt.hist(dat.cpu().detach().numpy(), bins=100)
plt.savefig('distribution.pdf')

