import torch
import torch.nn as nn

from uncertainty.predictive import predict_distribution


class TinyDropoutBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.cls = nn.Linear(1, 3)

    def forward(self, x, history, parameters=None):
        del parameters
        last = history[:, :, -1]
        prediction = last + self.dropout(torch.ones_like(last))
        pooled = prediction.mean(dim=1, keepdim=True)
        return prediction, self.cls(pooled)


def test_mc_dropout_returns_distribution_and_restores_mode():
    torch.manual_seed(4)
    model = TinyDropoutBackbone().train()
    x = torch.linspace(0, 1, 32).repeat(3, 1)
    history = torch.zeros(3, 32, 4)

    distribution = predict_distribution(model, x, history, mc_samples=16)
    assert model.training is True
    assert distribution.mean.shape == (3, 32)
    assert distribution.logits.shape == (3, 3)
    assert torch.all(distribution.variance >= 0)
    assert torch.any(distribution.variance > 0)
    assert torch.all(distribution.uncertainty > 0)

    deterministic = predict_distribution(model, x, history, mc_samples=1)
    assert torch.count_nonzero(deterministic.variance) == 0
