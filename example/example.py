import torch
import matplotlib.pyplot as plt
import itertools

from torch import nn
from helpers import gen_data, Mlp, plot_data
from tayloranalysis.model_extension import extend_model
from sklearn.model_selection import train_test_split

# load data and setup data
names = ["x1", "x2"]
x, y = gen_data(500)
plot_data(x, y)  # have a look at the data!

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.3)
x_test, y_test, x_train, y_train = map(
    lambda x: torch.tensor(x).float(),
    (x_test, y_test, x_train, y_train),
)

# one hot encode y train
y_train = nn.functional.one_hot(y_train.to(torch.int64), 3).float()

# initialize standard pytorch MLP model and wrap it with TaylorAnalysis
model = Mlp(input_neurons=2, hidden_neurons=10, output_neurons=3, hiddenlayers=2)
model = extend_model(model, reduce_function=torch.mean)

# setup for normal training
optim = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()

# choose device for training
device = torch.device(0)
x_train = x_train.to(device)
y_train = y_train.to(device)
model.to(device)

# specify the features for which the taylor coefficients should be calculated
combinations = [[0], [1]]  # 1st order taylor coefficients
combinations += [
    list(i) for i in itertools.permutations([0, 1], 2)  # 2nd order taylor coefficients
]
combinations += [[0, 1, 1]]  # exemplary 3rd order taylor coefficient

print(combinations)
tcs_training = []
x_train.requires_grad = True
for epoch in range(250):
    optim.zero_grad()
    pred = model(x_train)
    loss = crit(pred, y_train)
    loss.backward()
    optim.step()

    print("Epoch {}: Loss: {:.3f}".format(epoch + 1, loss))

    # get set of taylor coefficients during training
    tc_dict = model.get_tc(x_test.to(device), combinations, feature_names=names)
    tcs_training.append(list(tc_dict.values()))

# plot tcs during training
labels = list(tc_dict.keys())
plt.title("Taylor Coefficients during Training for given Features")
plt.plot(tcs_training, label=labels)
plt.xlabel("Epoch")
plt.ylabel("Taylor Coefficient Value")
plt.legend()
plt.savefig("training_evolution.png", bbox_inches="tight")
plt.clf()

# get a set of target taylor coefficients after training
model.cpu()
tc_dict = model.get_tc(x_test, combinations, feature_names=names)

# plot tcs after training
labels = list(tc_dict.keys())
plt.title("Taylor Coefficients after Training for given Features")
plt.plot(labels, list(tc_dict.values()), "+", color="black", markersize=10)
plt.xlabel("Taylor Coefficient")
plt.ylabel("Taylor Coefficient Value")
plt.savefig("taylor_coefficients.png", bbox_inches="tight")
plt.clf()
