import torch
from torch.utils.data import Dataset


class LineConceptDataset(Dataset):
    def __init__(self,B, T, N, DX=2, num_samples=10_000, device="cpu"):
        self.num_samples = num_samples
        self.device = device
        self.B = B
        self.T = T
        self.N = N
        self.DX = DX

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Each __getitem__ returns a fresh synthetic sample (concept instance)
        batch = self.sample_line_batch(device=self.device)
        return {k: v.squeeze(0) for k, v in batch.items()}  # remove batch dim
    
    def sample_line_batch(self, noise=0.02, length=1.6, k_attend=4, device="cpu"):
        '''
        Returns a dict with x0, x1, a (tensors).
        - x0: random initial positions in [-1,1]^2 at t=0, and near a line at t=1 for attended entities
        - a: real-valued attention where attended indices have larger positive values
        - x1: equal to x0 here (T=2 frames, last is the "after" state)
        '''
        x = torch.empty(self.B, self.T, self.N, self.DX, device=device)

        # t=0 random in [-1,1], t=1 start as copy
        x[:, 0] = torch.rand(self.B, self.N, self.DX, device=device) * 2 - 1
        x[:, 1] = x[:, 0].clone()

        a = torch.zeros(self.B, self.N, device=device)

        for b in range(self.B):
            idx = torch.randperm(self.N, device=device)[:k_attend]
            a[b, idx] = 3.0  # positive => sigmoid(a) ~ 0.95

            # construct a random line at t=1: p0 + t * dir
            p0 = torch.rand(2, device=device) * 2 - 1
            direction = torch.rand(2, device=device) * 2 - 1
            direction = direction / (direction.norm() + 1e-8)
            t_vals = torch.linspace(-length/2, length/2, k_attend, device=device)
            line_pts = p0 + t_vals[:, None] * direction
            x[b, 1, idx, :2] = line_pts + noise * torch.randn_like(line_pts)

            # non-attended can move slightly/randomly between frames
            others = torch.tensor([i for i in range(self.N) if i not in idx], device=device)
            x[b, 1, others, :2] = x[b, 0, others, :2] + 0.05 * torch.randn_like(x[b, 0, others, :2])

        return {"x0": x[:, :1].repeat(1, self.T, 1, 1),  # keep 2 frames for interface consistency
                "x1": x,
                "a": a}
    
    