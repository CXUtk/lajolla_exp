import torch
import torch.nn as nn

class RealNVPBlock(nn.Module):
    def __init__(self, dim, even):
        super(RealNVPBlock, self).__init__()
        self.dim = dim
        self.even = even
        # Determine the size of the partition
        self.d = dim // 2
        
        # Scale and translation networks, simple MLPs
        self.scale_net = nn.Sequential(
            nn.Linear(self.d, 64),
            nn.ReLU(),
            nn.Linear(64, self.d),
            nn.Tanh()  # Ensures the scaling operation is stable
        )
        self.translation_net = nn.Sequential(
            nn.Linear(self.d, 64),
            nn.ReLU(),
            nn.Linear(64, self.d)
        )

    def forward(self, x):
        # Partition the input
        if self.even:
            x1, x2 = x[:, 0::2], x[:, 1::2]
        else:
            x1, x2 = x[:, 1::2], x[:, 0::2]
        # x1, x2 = x[:, :self.d], x[:, self.d:]
        
        # Compute scale and translation parameters
        s = self.scale_net(x1)
        t = self.translation_net(x1)
        
        # Apply the affine transformation to x2
        y2 = torch.exp(s) * x2 + t
        # Concatenate the unchanged part with the transformed part
        y = torch.cat([x1, y2], dim=1)
        
        # Compute the log determinant of the Jacobian
        log_det_J = s.sum(dim=1)
        
        return y, log_det_J

    def inverse(self, y):
        # Partition the output
        if self.even:
            y1, y2 = y[:, 0::2], y[:, 1::2]
        else:
            y1, y2 = y[:, 1::2], y[:, 0::2]
        # y1, y2 = y[:, :self.d], y[:, self.d:]
        
        # Compute scale and translation parameters
        s = self.scale_net(y1)
        t = self.translation_net(y1)
        
        # Invert the affine transformation
        x2 = (y2 - t) * torch.exp(-s)
        # Concatenate to get the original x
        x = torch.cat([y1, x2], dim=1)
        
        return x

class RealNVP(nn.Module):
    def __init__(self, inputDim, layers=2):
        super().__init__()

        transforms = []
        for i in range(layers):
            transforms.append(RealNVPBlock(inputDim, bool(i % 2 == 0)))
        self.transforms = nn.ModuleList(transforms)
        
    def forward(self, x):
        log_det_jacobian = 0
        for transform in self.transforms:
            x, ldj = transform(x)
            log_det_jacobian += ldj
        return torch.sigmoid(x), log_det_jacobian

class AffineCoupling(nn.Module):
    def __init__(self, dim):
        super(AffineCoupling, self).__init__()
        self.scale = nn.Parameter(torch.randn(dim))
        self.shift = nn.Parameter(torch.randn(dim))
        
    def forward(self, x):
        y = torch.exp(self.scale) * x + self.shift
        log_det_jacobian = self.scale.sum()
        return y, log_det_jacobian
    
    def inverse(self, y):
        x = (y - self.shift) * torch.exp(-self.scale)
        return x

class NormalizingFlow(nn.Module):
    def __init__(self, base_dist, transforms):
        super(NormalizingFlow, self).__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        
    def forward(self, x):
        log_det_jacobian = 0
        for transform in self.transforms:
            x, ldj = transform(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian
    
    def sample(self, num_samples):
        z = self.base_dist.sample((num_samples,))
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z
    
    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        return self.base_dist.log_prob(z) + log_det_jacobian