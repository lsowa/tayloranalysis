import torch
import pickle
import sklearn

from torch import nn
from taylor import TaylorAnalysis

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

data = pickle.load(open("data/data.pickle", "rb"), encoding="latin-1")

x_train = torch.tensor(data['x_train'], dtype=torch.float)#[:10,:]
y_train = torch.tensor(data['y_train'], dtype=torch.float)#[:10]

model = Mlp(2, 100, 1, 1)
model = TaylorAnalysis(model)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.BCELoss()

device=torch.device(3)

x_train=x_train.to(device)
y_train=y_train.to(device)
model.to(device)


for epoch in range(200):
    optim.zero_grad()
    pred = model(x_train)
    loss = crit(pred, y_train)
    loss.backward()
    optim.step()
    print('Epoch {}: Loss: {:.3f}'.format(epoch+1, loss))
    model.tc_checkpoint(x_train, order=2)

del x_train, y_train

x_test = torch.tensor(data['x_test'], dtype=torch.float).to(device)

model.plot_tc(data=x_test, names=['x1', 'x2'], path='/work/lsowa/taylorcoefficients/', order=3)

#model.tc_plt_checkpoints(names=['x1', 'x2'], path='/work/lsowa/taylorcoefficients/')


