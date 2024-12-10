import torch
import unittest

from src.tayloranalysis import extend_model, BaseTaylorAnalysis


class Polynom(torch.nn.Module, BaseTaylorAnalysis):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, point):
        y = point[:, 1]
        x = point[:, 0]
        return x * y + x * y**2 + y**3


class FlippedPolynom(torch.nn.Module, BaseTaylorAnalysis):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, point):
        y = point[1, :]
        x = point[0, :]
        return x * y + x * y**2 + y**3


class TestTCComputation(unittest.TestCase):
    def setUp(self):
        # setup models
        self.models = [
            Polynom(),
            FlippedPolynom(),
        ]
        self.feature_axis = (-1, -2)

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
        self.points = [
            torch.tensor([[3, 2]]).float(),
            torch.tensor([[3, 2]]).float().t(),
        ]

    def test_coefficients(self):
        for combination in self.solution_dict.keys():
            for _model, _feature_axis, _point in zip(
                self.models,
                self.feature_axis,
                self.points,
            ):
                # compute TC
                tc = _model.get_tc(
                    "point",
                    forward_kwargs={"point": _point},
                    tc_idx_list=[combination],
                    tctensor_features_axis=_feature_axis,
                )
                tc = tc[combination].item()  # as float
                with self.subTest(combination=combination, feature_axis=_feature_axis):
                    # compare result to expected value
                    self.assertAlmostEqual(tc, self.solution_dict[combination])

    def test_coefficients_multithreading(self):
        for combination in self.solution_dict.keys():
            for _model, _feature_axis, _point in zip(
                self.models,
                self.feature_axis,
                self.points,
            ):
                # compute TC
                tc = _model.get_tc(
                    "point",
                    forward_kwargs={"point": _point},
                    tc_idx_list=[combination],
                    tctensor_features_axis=_feature_axis,
                    n_threads=3,
                )
                tc = tc[combination].item()  # as float
                with self.subTest(combination=combination, feature_axis=_feature_axis):
                    # compare result to expected value
                    self.assertAlmostEqual(tc, self.solution_dict[combination])


if __name__ == "__main__":
    # run TestTCComputation
    unittest.main()
