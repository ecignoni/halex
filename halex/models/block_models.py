from __future__ import annotations

import torch


class RidgeBlockModel(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, alpha: int = 1.0
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.alpha = alpha
        self.layer = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def regularization_loss(self, pred: torch.Tensor) -> torch.Tensor:
        # normalize by the number of samples
        return (
            self.alpha
            * torch.squeeze(self.layer.weight.T @ self.layer.weight)
            / pred.shape[0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
