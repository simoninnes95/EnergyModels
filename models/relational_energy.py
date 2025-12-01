import torch
import torch.nn as nn

def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [act()]
        elif out_act is not None:
            layers += [out_act()]
    return nn.Sequential(*layers)

class RelationalEnergy(nn.Module):
    '''
    E_theta(x, a, w) = [ f_theta( sum_{t,i,j} sig(a_i)*sig(a_j) * g_theta([x_t_i, x_t_j, w]), w ) ]^2
    Shapes:
      x: [B, T, N, DX]
      a: [B, N]     (real-valued; gated inside via sigmoid)
      w: [B, DW]
    returns energy: [B]
    '''
    def __init__(self, DX, DW, hidden=128):
        super().__init__()
        self.DX, self.DW = DX, DW
        self.g = mlp([2*DX + DW, hidden, hidden, hidden])
        self.f = mlp([hidden + DW, hidden, hidden, 1])

    def forward(self, x, a, w):
        B, T, N, DX = x.shape                            # [B, 2, 8, 2]
        sig_a = torch.sigmoid(a)                         # [B,N]
        m = sig_a.unsqueeze(2) * sig_a.unsqueeze(1)      # [B,N,N]
        m = m.unsqueeze(1).expand(B, T, N, N)            # [B,T,N,N]

        xi = x.unsqueeze(3).expand(B, T, N, N, DX)       # [B,T,N,N,DX]
        xj = x.unsqueeze(2).expand(B, T, N, N, DX)       # [B,T,N,N,DX]
        w_exp = w.view(B,1,1,1,-1).expand(B,T,N,N,-1)    # [B,T,N,N,DW] # -1 means not changing the size of this dim
        pair_feat = torch.cat([xi, xj, w_exp], dim=-1)   # [B,T,N,N,2DX+DW]

        g_ij = self.g(pair_feat)                         # [B,T,N,N,H]
        g_ij = g_ij * m.unsqueeze(-1)                    # gate by attention pairs
        pooled = g_ij.sum(dim=(1,2,3))                   # [B,H]

        f_in = torch.cat([pooled, w], dim=-1)            # [B,H+DW]
        out = self.f(f_in).squeeze(-1)                   # [B]
        energy = out.pow(2)                               # non-negative
        return energy