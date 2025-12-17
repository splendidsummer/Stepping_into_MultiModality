# Stepping_into_MultiModality
My ways to master multi-modal models 


# DiT impl
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Cosine diffusion schedule
# ----------------------------
def cosine_alpha(t):
    return torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

def q_sample(x0, t):
    noise = torch.randn_like(x0)
    alpha = cosine_alpha(t).view(-1, 1, 1)
    xt = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise
    return xt, noise

# ----------------------------
# Time Embedding (safe)
# ----------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

# ----------------------------
# Adaptive LayerNorm
# ----------------------------
class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]

# ----------------------------
# Transformer Block
# ----------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = AdaLayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = AdaLayerNorm(dim)

    def forward(self, x, cond):
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h

        h = self.norm2(x, cond)
        h = self.ffn(h)
        return x + h

# ----------------------------
# Diffusion Transformer Policy
# ----------------------------
class DiffusionTransformerPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon=8, dim=128, depth=6, heads=4):
        super().__init__()
        self.horizon = horizon

        self.obs_proj = nn.Linear(obs_dim, dim)
        self.act_proj = nn.Linear(action_dim, dim)

        # token type embedding: 0 = obs, 1 = action
        self.token_type = nn.Embedding(2, dim)

        self.time_emb = TimeEmbedding(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.out_proj = nn.Linear(dim, action_dim)

    def forward(self, obs, noisy_actions, t):
        B = obs.size(0)

        obs_token = self.obs_proj(obs) + self.token_type(
            torch.zeros(B, dtype=torch.long, device=obs.device)
        )

        act_tokens = self.act_proj(noisy_actions) + self.token_type(
            torch.ones(B, self.horizon, dtype=torch.long, device=obs.device)
        )

        x = torch.cat([obs_token.unsqueeze(1), act_tokens], dim=1)
        t_emb = self.time_emb(t)

        for block in self.blocks:
            x = block(x, t_emb)

        return self.out_proj(x[:, 1:])

# ----------------------------
# Train demo
# ----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionTransformerPolicy(
        obs_dim=32, action_dim=6, horizon=8
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3000):
        obs = torch.randn(16, 32).to(device)
        actions = torch.randn(16, 8, 6).to(device)
        t = torch.rand(16).to(device)

        xt, noise = q_sample(actions, t)
        pred = model(obs, xt, t)

        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 300 == 0:
            print(f"step {step} | loss={loss.item():.4f}")

if __name__ == "__main__":
    train()

```
