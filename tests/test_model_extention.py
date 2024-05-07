import unittest
import torch
import itertools

from src.tayloranalysis import extend_model

from torch import nn


class Mlp(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons, hidden_layers):
        nn.Module.__init__(self)

        # mlp layers
        self.linear1 = nn.Linear(input_neurons, output_neurons)
        self.linear2 = nn.Linear(output_neurons, output_neurons)
        # self.linear1.weight.data.fill_(0.1)
        # self.linear1.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        return x


mlp_specs = {
    "input_neurons": 5,
    "hidden_neurons": 10,
    "hidden_layers": 2,
    "output_neurons": 3,
}


x_data = torch.rand(10, 5)


class TestBaseClass:
    def test_forward(self):
        y = self.model(x_data)
        yshape = list(y.shape)
        targetshape = [x_data.shape[0], mlp_specs["output_neurons"]]
        self.assertEqual(yshape, targetshape)

    def test_node_compatibilities(self):
        # test if all-node results fit singe single-node and max-node results
        combinations = []
        for nth_order in [1, 2, 3]:
            combinations += [
                i
                for i in itertools.permutations(
                    range(mlp_specs["output_neurons"]), nth_order
                )
            ]
        for index in combinations:
            for eval_max_node_only in [False, True]:
                node_outputs = []
                for node in range(mlp_specs["output_neurons"]):
                    # get singe node results
                    tc = self.model.get_tc(
                        x_data=x_data,
                        index_list=[index],
                        node=node,
                        eval_max_node_only=eval_max_node_only,
                    )
                    tc = tc[index]
                    node_outputs.append(tc)
                # get all-node results
                tc = self.model.get_tc(
                    x_data=x_data,
                    index_list=[index],
                    node=None,
                    eval_max_node_only=eval_max_node_only,
                )
                tc = tc[index]
                # sums should be equal
                node_outputs = torch.stack(node_outputs, dim=-1).sum(dim=-1)
                with unittest.TestCase().subTest(
                    index=index, eval_max_node_only=eval_max_node_only
                ):
                    # check if tensors are close
                    is_close = torch.testing.assert_close(tc, node_outputs) == None
                    self.assertTrue(is_close)


class TestClassExtention(TestBaseClass, unittest.TestCase):
    def setUp(self):
        # check inheritance method
        WrappedModel = extend_model(Mlp)
        self.model = WrappedModel(**mlp_specs)


class TestInstanceExtention(TestBaseClass, unittest.TestCase):
    def setUp(self):
        # check adding methods to object
        model = Mlp(**mlp_specs)
        self.model = extend_model(model)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestClassExtention))
    suite.addTest(unittest.makeSuite(TestInstanceExtention))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    suite = test_suite()
    runner.run(suite)
