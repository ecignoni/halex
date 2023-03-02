import torch


class RidgeBlockModel(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.alpha = alpha
        self.layer = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, x):
        return self.layer(x)

    def regularization_loss(self, pred, y):
        # normalize by the number of samples
        return (
            self.alpha
            * torch.squeeze(self.layer.weight.T @ self.layer.weight)
            / pred.shape[0]
        )
