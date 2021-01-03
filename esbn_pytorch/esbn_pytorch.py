import torch
from torch import nn, einsum
from einops import repeat

# helpers

def exists(val):
    return val is not None

def safe_cat(t, el, dim = 0):
    if not exists(t):
        return el
    return torch.cat((t, el), dim = dim)

# classes

class ESBN(nn.Module):
    def __init__(
        self,
        *,
        value_dim = 64,
        key_dim = 64,
        hidden_dim = 512,
        output_dim = 4
    ):
        super().__init__()
        self.h0 = torch.zeros(1, 512)
        self.c0 = torch.zeros(1, 512)
        self.k0 = torch.zeros(1, key_dim + 1)

        self.rnn = nn.LSTMCell(key_dim + 1, hidden_dim)
        self.to_gate = nn.Linear(hidden_dim, 1)
        self.to_key = nn.Linear(hidden_dim, key_dim)
        self.to_output = nn.Linear(hidden_dim, output_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 4, stride = 2),
        )

        self.to_out = nn.Linear(4 * 64, value_dim)
        self.to_confidence = nn.Linear(1, 1)

    def forward(self, images):
        Mk = None
        Mv = None

        hx = self.h0
        cx = self.c0
        kx = self.k0
        out = []

        for ind, image in enumerate(images):
            is_first = ind == 0
            encoded = self.encoder(image)
            z = self.to_out(encoded.flatten(1))
            hx, cx = self.rnn(kx, (hx, cx))
            yt, gt, kwt = self.to_output(hx), self.to_gate(hx), self.to_key(hx)

            if is_first:
                kx = self.k0
            else:
                sim = einsum('b n d, b d -> b n', Mv, z)
                wkt = sim.softmax(dim = -1)
                ck = self.to_confidence(sim.unsqueeze(dim = -1)).sigmoid()
                kr = gt * (wkt.unsqueeze(-1) * torch.cat((Mk, ck), dim = -1)).sum(dim = 1)

            Mk = safe_cat(Mk, kwt.unsqueeze(1), dim = 1)
            Mv = safe_cat(Mv, z.unsqueeze(1), dim = 1)
            out.append(yt)

        return torch.stack(out)
