import torch
import torch.nn as nn
import torch.nn.functional as F
import code.config as C


class BiAGWrapper(nn.Module):
    def __init__(self, dim, depth=C.BIAG_DEPTH):
        super().__init__()
        self.biag = BiAG(dim, depth)

        self.dec_embed = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.kaiming_normal_(self.dec_embed, nonlinearity='relu')

        self.logit_alpha = nn.Parameter(torch.tensor(-3.0))

    def forward(self, p_new, p_old, w_old, use_alphs=False):
        if use_alphs:
            e_dir      = F.normalize(self.dec_embed, dim=-1)
            alpha      = F.softplus(self.logit_alpha)
            dec_offset = alpha * e_dir
            dec_raw =  p_new + dec_offset
        else:
            dec_raw = p_new

        dec     = F.layer_norm(dec_raw, dec_raw.shape[-1:])

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
    proj_mode = "post"  • Run attention in 2·D space, then      (original paper)
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
        self.scm  = SCM(dim)
        self.wsa  = WSA(dim)
        self.wpaa = WPAA(dim)
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, q, dec, p_old, w_old):
        W_s = self.scm(q)  # proto → weight
        W_s = self.wsa(W_s, dec)  # WSA
        q_P = self.scm(W_s)  # weight → proto
        new_w = self.wpaa(W_s, q_P, w_old, p_old)  # WPAA (q_P not updated q)
        q = q + self.gamma * self.scm(new_w)  # update AFTER WPAA, using SCM(new_w)
        return F.normalize(new_w, dim=-1), F.normalize(q, dim=-1)

class BiAG(nn.Module):
    def __init__(self, dim, depth=4, hidden=4 * 256):
        super().__init__()
        self.blocks = nn.ModuleList(BiAGBlock(dim) for _ in range(depth))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, p_new, p_old, w_old, dec_embed):
        """
        p_new : (B , Nn , D)
        p_old : (B , No , D)
        w_old : (B , No , D)
        dec_embed : (B , 1 , D)
        returns  (Nn , D)   –– normalised generated weights for new classes
        """
        q = p_new
        for blk in self.blocks:
            new_w, q = blk(q, dec_embed, p_old, w_old)
            q = F.normalize(self.mlp(q), dim=-1)

        # new_w is (B , Nn , D); episodes are batched one-at-a-time (B=1)
        new_w = new_w.squeeze(0)        # → (Nn , D)
        return F.normalize(new_w, dim=-1)
