import pytorch_lightning as pl
from torch import nn
import torch


class DLA(pl.LightningModule):
    def __init__(self, lighten, num_features=None, num_snapshots=100, hidden_size=128):
        super().__init__()
        self.name = "mlp"
        # Feature width depends on the data representation; default to the raw-LOB widths
        # (40 full / 20 lighten) when not provided. For pure OFI this is 10 / 5.
        if lighten:
            self.name += "-lighten"
        if num_features is None:
            num_features = 20 if lighten else 40

        self.W1 = nn.Linear(num_features, num_features, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )

        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(num_snapshots*hidden_size, 3)

    def forward(self, x):
        # x.shape = [batch_size, num_snapshots, num_features]
        x = x.squeeze(1)

        X_tilde = self.W1(x)
        # alpha.shape = [batch_size, num_snapshots, num_features]

        alpha = self.softmax(X_tilde)
        # alpha.shape = [batch_size, num_snapshots, num_features]

        alpha = torch.mean(alpha, dim=2)
        # alpha.shape = [batch_size, num_snapshots]

        x_tilde = torch.einsum('ij,ijk->ijk', [alpha, x])
        # x_tilde.shape = [batch_size, num_snapshots, num_features]

        H, _ = self.gru(x_tilde)
        # o.shape = [batch_size, num_snapshots, hidden_size]

        H_tilde = self.W2(H)
        # o.shape = [batch_size, num_snapshots, hidden_size]

        beta = self.softmax(H_tilde)
        # o.shape = [batch_size, num_snapshots, hidden_size]

        beta = torch.mean(beta, dim=2)
        # beta.shape = [batch_size, num_snapshots]

        h_tilde = torch.einsum('ij,ijk->ijk', [beta, H])
        # h_tilde.shape = [batch_size, num_snapshots, hidden_size]

        h_tilde = torch.flatten(h_tilde, start_dim=1)
        # h_tilde.shape = [batch_size, hidden_size*num_snapshots]

        logits = self.W3(h_tilde)
        # out.shape = [batch_size, 3]

        return logits