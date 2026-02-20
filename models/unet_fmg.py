import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -torch.log(torch.tensor(10000.0, device=device)), half, device=device))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        h = self.heads
        q = q.view(B, h, C // h, H * W)
        k = k.view(B, h, C // h, H * W)
        v = v.view(B, h, C // h, H * W)
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) / ((C // h) ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, H, W)
        return self.proj(out)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.gelu(self.norm1(x)))
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.gelu(self.norm2(h)))
        return h + self.skip(x)

class FMG(nn.Module):
    def __init__(self, channels: int, mem_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.k = nn.Linear(mem_dim, channels, bias=False)
        self.v = nn.Linear(mem_dim, channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, Ft):
        B, C, H, W = x.shape
        h = self.heads
        q = self.q(x).view(B, h, C // h, H * W).transpose(-1, -2)
        k = self.k(Ft).view(B, -1, h, C // h).transpose(1, 2)
        v = self.v(Ft).view(B, -1, h, C // h).transpose(1, 2)
        attn = torch.einsum("bhnc,bhkc->bhnk", q, k) / ((C // h) ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnk,bhkc->bhnc", attn, v)
        out = out.transpose(-1, -2).contiguous().view(B, C, H, W)
        out = self.proj(out)
        return x + out

class FMGBlock(nn.Module):
    def __init__(self, ch: int, mem_dim: int, t_dim: int, heads: int, use_sa: bool):
        super().__init__()
        self.res = ResBlock(ch, ch, t_dim)
        self.sa = SelfAttention2d(ch, heads) if use_sa else None
        self.fmg = FMG(ch, mem_dim, heads)

    def forward(self, x, t_emb, Ft):
        h = self.res(x, t_emb)
        if self.sa is not None:
            h = self.sa(h)
        h = self.fmg(h, Ft)
        return h

class UNetWithFMG(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, mem_dim: int, heads: int, levels: int):
        super().__init__()
        self.levels = levels
        t_dim = base_ch
        self.time = SinusoidalTimeEmbedding(t_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.GELU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        chs = [base_ch * (2 ** i) for i in range(levels)]
        self.down_blocks1 = nn.ModuleList()
        self.down_blocks2 = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(levels):
            ch = chs[i]
            if i == 0:
                self.down_blocks1.append(FMGBlock(ch, mem_dim, t_dim, heads, use_sa=False))
                self.down_blocks2.append(FMGBlock(ch, mem_dim, t_dim, heads, use_sa=True))
            else:
                self.down_blocks1.append(FMGBlock(ch, mem_dim, t_dim, heads, use_sa=False))
                self.down_blocks2.append(FMGBlock(ch, mem_dim, t_dim, heads, use_sa=True))
            if i < levels - 1:
                self.downs.append(nn.Conv2d(ch, chs[i + 1], 4, stride=2, padding=1))

        mid_ch = chs[-1]
        self.mid1 = FMGBlock(mid_ch, mem_dim, t_dim, heads, use_sa=True)
        self.mid2 = FMGBlock(mid_ch, mem_dim, t_dim, heads, use_sa=True)

        self.ups = nn.ModuleList()
        self.up_blocks1 = nn.ModuleList()
        self.up_blocks2 = nn.ModuleList()

        for i in range(levels - 1, -1, -1):
            ch = chs[i]
            if i < levels - 1:
                self.ups.append(nn.ConvTranspose2d(chs[i + 1], ch, 4, stride=2, padding=1))
            in_cat = ch * 2 if i < levels - 1 else ch * 2
            self.up_blocks1.append(ResBlock(in_cat, ch, t_dim))
            self.up_blocks2.append(FMGBlock(ch, mem_dim, t_dim, heads, use_sa=True))

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, Ft):
        t_emb = self.t_mlp(self.time(t))
        h = self.in_conv(x)

        skips = []
        for i in range(self.levels):
            h = self.down_blocks1[i](h, t_emb, Ft)
            h = self.down_blocks2[i](h, t_emb, Ft)
            skips.append(h)
            if i < self.levels - 1:
                h = self.downs[i](h)

        h = self.mid1(h, t_emb, Ft)
        h = self.mid2(h, t_emb, Ft)

        for i in range(self.levels - 1, -1, -1):
            if i < self.levels - 1:
                up = self.ups[self.levels - 2 - i](h)
            else:
                up = h
            skip = skips[i]
            cat = torch.cat([up, skip], dim=1)
            h = self.up_blocks1[self.levels - 1 - i](cat, t_emb)
            h = self.up_blocks2[self.levels - 1 - i](h, t_emb, Ft)

        out = self.out_conv(F.gelu(self.out_norm(h)))
        return out