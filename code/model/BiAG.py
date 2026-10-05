import torch
import torch.nn as nn
import torch.nn.functional as F
import code.config as C


class BiAGWrapper(nn.Module):
    def __init__(self, dim, depth=None):
        super().__init__()
        self.biag = BiAG(dim, C.BIAG_DEPTH if depth is None else depth)

    def forward(self, p_new, p_old, w_old):
        # A single episode has B=1 and one token per new/old class.
        if any(x.ndim != 3 or x.size(0) != 1 for x in (p_new, p_old, w_old)):
            raise ValueError("BiAG expects one episode: (1, new_classes, D), (1, old_classes, D)")
        if p_old.shape != w_old.shape or p_new.size(-1) != p_old.size(-1):
            raise ValueError("Prototype and weight feature dimensions must match")
        if p_new.size(1) == 0 or p_old.size(1) == 0:
            raise ValueError("BiAG requires nonempty new and old class sets")
        # Preserve the original default decoder initialization. The paper does
        # not fully specify d_E initialization; this is an implementation choice.
        dec = F.layer_norm(p_new, p_new.shape[-1:])
        return self.biag(p_new, p_old, w_old, dec)

class SCM(nn.Module):
    """Semantic Conversion Module (proto ⇆ weight)."""
    def __init__(self, dim, hidden=4 * 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)      # keep on unit sphere

class WSA(nn.Module):
    """
    Weight Self-Attention.
    Q = K = q_w + dE    (element-wise add)
    V = dE              (decoder embeddings)
    Shapes are batch-first: (B , L , D)
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, q_w, dE):
        """
        q_w : (B , Lq , D)
        dE  : (1 or B , Ld=1 , D)
        """
        # broadcast decoder embeddings if they were stored as (1 , 1 , D)
        if dE.size(0) == 1 and q_w.size(0) > 1:
            dE = dE.expand(q_w.size(0), -1, -1)
        elif dE.size(0) != q_w.size(0):
            raise ValueError("Batch size of dE must be 1 or equal to q_w")

        q_w = F.normalize(q_w, dim=-1)
        dE = F.normalize(dE, dim=-1)
        fused = q_w + dE                       # (B , Lq , D)
        out, _ = self.attn(fused, fused, dE, average_attn_weights=False)
        return out                             # (B , Lq , D)

class WPAA(nn.Module):
    """
    Weight & Prototype Analogical Attention (eqs. 10-15)

    Two projection variants
    -----------------------
    proj_mode = "pre"   • Pre-project 2·D → D before attention  (compact)
    proj_mode = "post"  • Run attention in 2·D space, then
                          project 2·D → D afterwards
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        proj_mode: str = "post"        # "pre"  or  "post"
    ):
        super().__init__()
        assert proj_mode in ("pre", "post"), \
            "proj_mode must be 'pre' or 'post'"
        self.proj_mode = proj_mode
        self.dim = dim
        # self.log_tau = nn.Parameter(torch.tensor(0.5))  # exp(1)=2.72

        if proj_mode == "pre":
            self.qk_proj = nn.Linear(2 * dim, dim, bias=False)
            self.cross = nn.MultiheadAttention(
                embed_dim=dim, num_heads=8, dropout=0.1,
                batch_first=True)
        else:  # "post"
            self.cross = nn.MultiheadAttention(
                embed_dim=2 * dim,        # Q / K / output width
                num_heads=num_heads,
                kdim=2 * dim,
                vdim=dim,
                dropout=dropout,
                batch_first=True,
            )
            # 2·D → D AFTER attention
            self.out_proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(
        self,
        W_s: torch.Tensor,    # (B , Nn , D)
        q_P: torch.Tensor,    # (B , Nn , D)
        W_old: torch.Tensor,  # (B , No , D)
        p_old: torch.Tensor   # (B , No , D)
    ) -> torch.Tensor:        # (B , Nn , D)

        W_s = F.normalize(W_s, dim=-1)
        q_P = F.normalize(q_P, dim=-1)
        W_old = F.normalize(W_old, dim=-1)
        p_old = F.normalize(p_old, dim=-1)

        if self.proj_mode == "pre":   # pre-projection
            Q_c = self.qk_proj(torch.cat([W_s, q_P], dim=-1))  # (B , Nn , D)
            K_c = self.qk_proj(torch.cat([W_old, p_old], dim=-1))  # (B , No , D)
            V_c = W_old                                           # (B , No , D)
            out, attn = self.cross(Q_c, K_c, V_c)                    # (B , Nn , D)
            self.last_attn = attn  # for debug
            return F.normalize(out, dim=-1)
        else:
            Q_c = torch.cat([W_s,  q_P],  dim=-1)  # (B , Nn , 2·D)
            K_c = torch.cat([W_old, p_old], dim=-1)# (B , No , 2·D)
            # tau = torch.clamp(self.log_tau.exp(), max=10.0)
            # Q_c = torch.cat([W_s,  q_P],  dim=-1)
            # K_c = torch.cat([W_old, p_old], dim=-1) * tau
            V_c = W_old                             # (B , No ,   D)
            out, attn = self.cross(Q_c, K_c, V_c, average_attn_weights=False)  # (B , Nn , 2·D)
            self.last_attn = attn    # for debug
            out = self.out_proj(out)                # (B , Nn ,   D)
            return out

class BiAGBlock(nn.Module):
    """A single reasoning layer."""
    def __init__(self, dim):
        super().__init__()
        self.wsa  = WSA(dim)
        self.wpaa = WPAA(dim)

    def forward(self, q, dec, p_old, w_old, scm):
        converted_q = scm(q)
        W_s = self.wsa(converted_q, dec)
        # Both semantic branches originate from q_L (paper eqs. 7 and 10).
        new_w = self.wpaa(W_s, converted_q, w_old, p_old)
        q = q + scm(new_w)  # paper eq. 16, with SCM shared across layers
        return F.normalize(new_w, dim=-1), q

class BiAG(nn.Module):
    def __init__(self, dim, depth=4, hidden=4 * 256):
        super().__init__()
        if depth < 1:
            raise ValueError("BiAG depth must be positive")
        self.scm = SCM(dim, hidden)
        self.blocks = nn.ModuleList(BiAGBlock(dim) for _ in range(depth))

    def forward(self, p_new, p_old, w_old, dec_embed):
        """
        p_new : (B , Nn , D)
        p_old : (B , No , D)
        w_old : (B , No , D)
        dec_embed : (1 , Nn , D)
        returns  (Nn , D)   –– normalised generated weights for new classes
        """
        q = p_new
        for blk in self.blocks:
            new_w, q = blk(q, dec_embed, p_old, w_old, self.scm)

        # new_w is (B , Nn , D); episodes are batched one-at-a-time (B=1)
        new_w = new_w.squeeze(0)        # → (Nn , D)
        return F.normalize(new_w, dim=-1)
