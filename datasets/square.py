import torch
from torch.utils.data import Dataset


class SquareConceptDataset(Dataset):
    def __init__(self, B, T, N, DX=2, num_samples=10_000, device="cpu"):
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
        batch = self.sample_square_batch(device=self.device)
        return {k: v.squeeze(0) for k, v in batch.items()}  # remove batch dim
    
    def sample_square_batch(self, noise=0.02, side_length=1.2, k_attend=4, device="cpu"):
        '''
        Returns a dict with x0, x1, a (tensors).
        - x0: random initial positions in [-1,1]^2 at t=0, and arranged in a square at t=1 for attended entities
        - a: real-valued attention where attended indices have larger positive values
        - x1: equal to x0 at t=0, and square positions at t=1
        '''
        x = torch.empty(self.B, self.T, self.N, self.DX, device=device)

        # t=0 random in [-1,1], t=1 start as copy
        x[:, 0] = torch.rand(self.B, self.N, self.DX, device=device) * 2 - 1
        x[:, 1] = x[:, 0].clone()

        a = torch.zeros(self.B, self.N, device=device)

        for b in range(self.B):
            idx = torch.randperm(self.N, device=device)[:k_attend]
            a[b, idx] = 3.0  # positive => sigmoid(a) ~ 0.95

            # construct a random square at t=1
            # center of the square
            center = torch.rand(2, device=device) * 2 - 1
            
            # random rotation angle
            angle = torch.rand(1, device=device).item() * 2 * 3.14159
            cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
            
            # square corners in canonical position (before rotation)
            half_side = side_length / 2
            corners = torch.tensor([
                [-half_side, -half_side],
                [half_side, -half_side],
                [half_side, half_side],
                [-half_side, half_side]
            ], device=device)
            
            # rotate the corners
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ], device=device)
            
            rotated_corners = torch.matmul(corners, rotation_matrix.T)
            
            # translate to center
            square_pts = rotated_corners + center
            
            # distribute k_attend entities around the square perimeter
            if k_attend <= 4:
                # place on corners
                selected_pts = square_pts[:k_attend]
            else:
                # distribute evenly around perimeter
                perimeter_positions = torch.linspace(0, 4, k_attend, device=device)
                selected_pts = torch.zeros(k_attend, 2, device=device)
                
                for i, pos in enumerate(perimeter_positions):
                    # which side (0-3)
                    side = int(pos.item())
                    t = pos.item() - side  # position along that side [0,1]
                    
                    if side == 0:  # bottom
                        pt = square_pts[0] * (1 - t) + square_pts[1] * t
                    elif side == 1:  # right
                        pt = square_pts[1] * (1 - t) + square_pts[2] * t
                    elif side == 2:  # top
                        pt = square_pts[2] * (1 - t) + square_pts[3] * t
                    else:  # left (side == 3)
                        pt = square_pts[3] * (1 - t) + square_pts[0] * t
                    
                    selected_pts[i] = pt
            
            x[b, 1, idx, :2] = selected_pts[:k_attend] + noise * torch.randn(k_attend, 2, device=device)

            # non-attended entities move slightly/randomly between frames
            others = torch.tensor([i for i in range(self.N) if i not in idx], device=device)
            if len(others) > 0:
                x[b, 1, others, :2] = x[b, 0, others, :2] + 0.05 * torch.randn_like(x[b, 0, others, :2])

        return {"x0": x[:, :1].repeat(1, self.T, 1, 1),  # keep 2 frames for interface consistency
                "x1": x,
                "a": a}
