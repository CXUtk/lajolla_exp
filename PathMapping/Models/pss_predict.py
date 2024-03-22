import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        if m.bias != None:
            torch.nn.init.normal_(m.bias.data, std=0.1) #xavier not applicable for biases

#ToDO Fill in the __ values
class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 3)
        )

        self.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)  # size=(N, 3)
