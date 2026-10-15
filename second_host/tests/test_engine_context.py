import unittest

from _bootstrap import ROOT  # noqa: F401
import torch
from torch import nn

from src.solver.det_engine import train_one_epoch


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, samples, targets):
        return {"value": samples * self.weight}


class ContextCriterion(nn.Module):
    requires_step_context = True

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, outputs, targets, **context):
        self.seen.append(context)
        return {"tiny_loss": outputs["value"].sum()}


class EngineContextTest(unittest.TestCase):
    def test_epoch_and_step_are_forwarded_only_to_opted_in_criterion(self):
        model = TinyModel()
        criterion = ContextCriterion()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = [
            (torch.ones(1), [{"label": torch.tensor(0)}]),
            (torch.ones(1), [{"label": torch.tensor(1)}]),
        ]
        train_one_epoch(
            model,
            criterion,
            data,
            optimizer,
            torch.device("cpu"),
            epoch=3,
            print_freq=100,
        )
        self.assertEqual(len(criterion.seen), 2)
        self.assertEqual(criterion.seen[0], {
            "epoch": 3, "step": 0, "epoch_step": 2, "global_step": 6
        })
        self.assertEqual(criterion.seen[1], {
            "epoch": 3, "step": 1, "epoch_step": 2, "global_step": 7
        })


if __name__ == "__main__":
    unittest.main()
