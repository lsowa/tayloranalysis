import torch
from torch.func import grad, vmap
from torch import nn


class Mlp(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, hiddenlayers):

        nn.Module.__init__(self)

        # mlp layers
        self.mlplayers = nn.ModuleList([nn.Linear(input_neurons, hidden_neurons)])
        self.mlplayers.extend(
            [nn.Linear(hidden_neurons, hidden_neurons) for i in range(hiddenlayers + 1)]
        )
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
        return x[0], x[1]


model = Mlp(10, 100, 2, 2)

x = torch.randn(3, 10, requires_grad=True)
print("x shape", x.shape)
y = model(x)
print("y ", y)
mfunc = vmap(grad(model, has_aux=True))(x)
print(mfunc)
