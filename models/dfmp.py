import torch
import torch.nn as nn
import torch.nn.functional as F

class DFMP(nn.Module):
    """
    Diffusion Fuzzy Memory Particles (DFMP)

    Key fixes vs. your original:
    - defuzzify_centroid(): r_tilde is no longer treated as constant memberships on grid.
      We map rb/rs to per-dimension fuzzy-set location parameters so centroid depends on r_tilde.
    - forward(): fixed the fuzzify call bug (self.fuzzify_vec(lr) / self.fuzzify_vec(g)).
    """

    def __init__(self, lr_dim: int, g_dim: int, mem_particles: int, mem_dim: int, T: int):
        super().__init__()
        self.K = mem_particles
        self.d = mem_dim
        self.T = T

        self.FT = nn.Parameter(torch.randn(self.K, self.d) * 0.02)

        self.proj_lr = nn.Linear(lr_dim, self.d)
        self.proj_g = nn.Linear(g_dim, self.d)

        # q takes 2d (fuzzy concatenation), outputs d
        self.Wq_I = nn.Linear(2 * self.d, self.d, bias=False)
        self.Wq_G = nn.Linear(2 * self.d, self.d, bias=False)

        # k takes d (memory particle), outputs d
        self.Wk_I = nn.Linear(self.d, self.d, bias=False)
        self.Wk_G = nn.Linear(self.d, self.d, bias=False)

        # v takes d, outputs 2d (fuzzy descriptor)
        self.Wv_I = nn.Linear(self.d, 2 * self.d, bias=False)
        self.Wv_G = nn.Linear(self.d, 2 * self.d, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(self.d, self.d * 2),
            nn.GELU(),
            nn.Linear(self.d * 2, self.d),
        )

        self.Wf = nn.Linear(self.d, self.d, bias=True)
        self.Wi = nn.Linear(self.d, self.d, bias=True)
        self.Vf = nn.Linear(self.d, self.d, bias=False)
        self.Vi = nn.Linear(self.d, self.d, bias=False)

        # defuzzify grid
        grid = torch.linspace(-4.0, 4.0, 41)  # (G,)
        self.register_buffer("grid", grid)

        # global bell-shape parameters (learnable)
        self.a_bell = nn.Parameter(torch.tensor(1.0))
        self.b_bell = nn.Parameter(torch.tensor(2.0))

        # NOTE:
        # We will NOT use a fixed global center c for bell in defuzzify;
        # we treat rb as per-dimension center (after squashing).
        # For S-function, we treat rs as per-dimension center and build (a,b) around it.

    # -------- fuzzy membership functions (vectorized) --------
    def mu_bell_param(self, x: torch.Tensor, c: torch.Tensor):
        """
        Generalized bell membership with per-element center c.
        x: (..., G)
        c: (..., 1) broadcastable
        """
        a = torch.clamp(self.a_bell, 1e-3)
        b = torch.clamp(self.b_bell, 1e-3)
        return 1.0 / (1.0 + torch.abs((x - c) / a) ** (2.0 * b))

    @staticmethod
    def mu_s_param(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        """
        Smooth S-shaped membership with per-element (a,b).
        x: (..., G)
        a,b: (..., 1) broadcastable, expected a < b
        """
        eps = 1e-12
        a2 = torch.minimum(a, b)
        b2 = torch.maximum(a, b)
        mid = (a2 + b2) / 2.0

        y = torch.zeros_like(x)
        # (a, mid]
        y = torch.where(
            (x > a2) & (x <= mid),
            2.0 * ((x - a2) / (b2 - a2 + eps)) ** 2,
            y,
        )
        # (mid, b]
        y = torch.where(
            (x > mid) & (x <= b2),
            1.0 - 2.0 * ((b2 - x) / (b2 - a2 + eps)) ** 2,
            y,
        )
        # > b
        y = torch.where(x > b2, torch.ones_like(x), y)
        return y

    def fuzzify_vec(self, x: torch.Tensor):
        """
        x: (B, d) -> fuzzy features (B, 2d)
        Here we use fixed-shape membership functions applied to x directly.
        This is OK as long as x is properly scaled by proj layers.
        """
        # For fuzzification, we interpret x itself as the variable
        # and compute membership degrees for bell / S.
        # To keep it stable, squash x into roughly the grid range.
        x_squash = 4.0 * torch.tanh(x / 4.0)
        # bell centered at 0 for fuzzification (symmetric)
        xb = self.mu_bell_param(x_squash, c=torch.zeros_like(x_squash))
        # S-function around 0 -> we set a=-1, b=1 (fixed) for fuzzification step
        a = (-1.0) * torch.ones_like(x_squash)
        b = ( 1.0) * torch.ones_like(x_squash)
        xs = self.mu_s_param(x_squash, a=a, b=b)
        return torch.cat([xb, xs], dim=-1)

    def defuzzify_centroid(self, r_tilde_2d: torch.Tensor):
        """
        r_tilde_2d: (B, 2d) produced by Wv(*)

        FIX:
        - Original code expanded rb/rs as constant membership weights over grid -> centroid ~= mean(grid) ~= 0.
        - Now we interpret rb/rs as per-dimension *location parameters* of fuzzy sets on the grid:
            rb -> bell center c_b
            rs -> S center c_s, and we construct (a_s, b_s) = (c_s - delta, c_s + delta)
        """
        B = r_tilde_2d.shape[0]
        d = r_tilde_2d.shape[1] // 2

        rb = r_tilde_2d[:, :d]   # (B, d)
        rs = r_tilde_2d[:, d:]   # (B, d)

        # grid: (1, 1, G) -> (B, d, G)
        g = self.grid.view(1, 1, -1).expand(B, d, -1)

        # map rb/rs into grid range [-4, 4] to stabilize
        c_b = 4.0 * torch.tanh(rb).unsqueeze(-1)  # (B, d, 1)
        c_s = 4.0 * torch.tanh(rs).unsqueeze(-1)  # (B, d, 1)

        # bell membership centered at c_b
        mu_b = self.mu_bell_param(g, c=c_b)       # (B, d, G)

        # S membership centered around c_s with width delta
        delta = 1.0
        a_s = torch.clamp(c_s - delta, -4.0, 4.0)
        b_s = torch.clamp(c_s + delta, -4.0, 4.0)
        mu_s = self.mu_s_param(g, a=a_s, b=b_s)   # (B, d, G)

        # centroid for each fuzzy set
        cen_b = (g * mu_b).sum(dim=-1) / (mu_b.sum(dim=-1) + 1e-12)  # (B, d)
        cen_s = (g * mu_s).sum(dim=-1) / (mu_s.sum(dim=-1) + 1e-12)  # (B, d)

        # simple fusion (you can also learn weights)
        return 0.5 * cen_b + 0.5 * cen_s  # (B, d)

    def _fuzzy_distance(self, q_fuzzy_2d: torch.Tensor, Ft: torch.Tensor, Wq: nn.Linear, Wk: nn.Linear):
        """
        q_fuzzy_2d: (B, 2d)
        Ft: (B, K, d)
        """
        Q = Wq(q_fuzzy_2d)  # (B, d)
        K = Wk(Ft)          # (B, K, d)
        scores = torch.einsum("bd,bkd->bk", Q, K) / (self.d ** 0.5)
        return scores

    def _memory_response(self, scores: torch.Tensor, Ft: torch.Tensor, Wv: nn.Linear):
        """
        scores: (B, K)
        Ft: (B, K, d)
        returns r_tilde: (B, 2d)
        """
        W = F.softmax(scores, dim=-1)     # (B, K)
        V = Wv(Ft)                        # (B, K, 2d)
        r_tilde = torch.einsum("bk,bkd->bd", W, V)
        return r_tilde

    def _gated_update(self, Ft_next: torch.Tensor, r: torch.Tensor):
        """
        Ft_next: (B, K, d)
        r: (B, d)
        """
        B, K, d = Ft_next.shape
        r_b = r.unsqueeze(1).expand(B, K, d)

        F_tilde = self.mlp(Ft_next + r_b) + Ft_next + r_b
        Xf = self.Wf(r_b) + torch.tanh(self.Vf(Ft_next))
        Xi = self.Wi(r_b) + torch.tanh(self.Vi(Ft_next))

        f_gate = torch.sigmoid(Xf)
        i_gate = torch.sigmoid(Xi)

        Ft = f_gate * Ft_next + i_gate * torch.tanh(F_tilde)
        return Ft

    def forward(self, wsi_lr: torch.Tensor, genomics: torch.Tensor):
        """
        wsi_lr: (B, lr_dim)
        genomics: (B, g_dim)
        returns Ft_steps: list length T, each (B, K, d)
        """
        B = wsi_lr.shape[0]
        lr = self.proj_lr(wsi_lr)         # (B, d)
        g = self.proj_g(genomics)         # (B, d)

        # FIX: correct function call
        lr_fuzzy = self.fuzzify_vec(lr)   # (B, 2d)
        g_fuzzy = self.fuzzify_vec(g)     # (B, 2d)

        Ft_next = self.FT.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, K, d)
        Ft_steps = [None] * self.T

        # reverse step as your original (t = T..1)
        for step in range(self.T, 0, -1):
            scores_I = self._fuzzy_distance(lr_fuzzy, Ft_next, self.Wq_I, self.Wk_I)
            scores_G = self._fuzzy_distance(g_fuzzy, Ft_next, self.Wq_G, self.Wk_G)

            rtilde_I = self._memory_response(scores_I, Ft_next, self.Wv_I)  # (B, 2d)
            rtilde_G = self._memory_response(scores_G, Ft_next, self.Wv_G)  # (B, 2d)

            r_I = self.defuzzify_centroid(rtilde_I)  # (B, d)
            r_G = self.defuzzify_centroid(rtilde_G)  # (B, d)

            r = 0.5 * r_I + 0.5 * r_G               # (B, d)

            Ft = self._gated_update(Ft_next, r)      # (B, K, d)
            Ft_steps[step - 1] = Ft
            Ft_next = Ft

        return Ft_steps