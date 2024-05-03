import torch
import unittest

from src.tayloranalysis import extend_model


class Polynom(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, point):
        y = point[:, 1]
        x = point[:, 0]
        return x * y + x * y**2 + y**3


class TestTCComputation(unittest.TestCase):
    def setUp(self):
        # setup model
        WrappedModel = extend_model(Polynom)
        self.model = WrappedModel()

        # setup feature combinations and expected results
        self.solution_dict = {
            (0,): 6.0,
            (1,): 27.0,
            (0, 0): 0.0,
            (0, 1): 5.0,
            (1, 0): 5.0,
            (1, 1): 9.0,
            (0, 0, 0): 0.0,
            (0, 0, 1): 0.0,
            (0, 1, 0): 0.0,
            (0, 1, 1): 1.0,
            (1, 0, 0): 0.0,
            (1, 0, 1): 1.0,
            (1, 1, 0): 1.0,
            (1, 1, 1): 1.0,
        }
        # setup data to evaluate
        self.point = torch.tensor([[3, 2]]).float()

    def test_coefficients(self):
        for combination in self.solution_dict.keys():
            # compute TC
            tc = self.model.get_tc(
                x_data=self.point,
                index_list=[combination],
            )
            tc = tc[combination].item()  # as float
            with self.subTest(combination=combination):
                # compare result to expected value
                self.assertAlmostEqual(tc, self.solution_dict[combination])


if __name__ == "__main__":
    # run TestTCComputation
    unittest.main()
