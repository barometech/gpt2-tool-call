"""STEERING ADAPTER v2 — РАСШИРЕНИЕ существующего classifier'а.

Адаптер сверху + руль снизу:
  ВЕРХ (как было):
    h6 → bottle 192 → 96 → action(53)/scope/format/spec/target/ptr_s/ptr_e/gate
    Решает: ЧТО юзер хочет (action) и НАДО ли вызывать (gate).
  НИЗ (новое):
    тот же 96-dim pool → W_steer (96→768) → delta
    delta добавляется в hidden L6 → влияет на L7..L11 → vocab_logits смещаются

Один forward, оба уровня. Один backward — combined loss:
  L = CE(action) + BCE(gate) + λ * CE(vocab_logits[-1], gold_first_token)

Resume strategy:
  Загружаем classifier из torch_EN_BFCL.npz (все веса inherit)
  W_steer.zero_() — начало = чистый GPT-2 + classifier (как сейчас)

Это РАСШИРЕНИЕ — не ломает hard_cases 27/27 и BFCL 68%.
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, "code")

from integrated_gpt2_torch import GPT2, load_gpt2_torch_weights, encode, decode
from modes_spec_v5 import ACTIONS, SCOPES, FORMATS, SPECIFICITIES, TARGETS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullSteeringAdapter(nn.Module):
    """Все classifier heads (как было) + W_steer (новое)."""
    def __init__(self, d_hidden=768, d_bottle=192, d_out=96):
        super().__init__()
        # Bottle (как было)
        self.W1 = nn.Linear(d_hidden, d_bottle)
        self.ln1 = nn.LayerNorm(d_bottle)
        self.W2 = nn.Linear(d_bottle, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        # Classifier heads (как в torch_EN_BFCL)
        self.h_action = nn.Linear(d_out, len(ACTIONS))
        self.h_scope  = nn.Linear(d_out, len(SCOPES))
        self.h_format = nn.Linear(d_out, len(FORMATS))
        self.h_spec   = nn.Linear(d_out, len(SPECIFICITIES))
        self.h_target = nn.Linear(d_out, len(TARGETS))
        self.ptr_s    = nn.Linear(d_out, 1)
        self.ptr_e    = nn.Linear(d_out, 1)
        self.gate     = nn.Linear(d_out, 1)
        # NEW: steering — pool → 768 delta (per-batch, broadcast over T)
        self.W_steer  = nn.Linear(d_out, d_hidden)
        # Init steering to zero — pipeline = identity до обучения
        with torch.no_grad():
            self.W_steer.weight.zero_()
            self.W_steer.bias.zero_()

    def forward(self, h, mask):
        h1 = F.gelu(self.ln1(self.W1(h)))
        h2 = F.gelu(self.ln2(self.W2(h1)))               # (B, T, d_out)
        m = mask.unsqueeze(-1)
        pooled = (h2 * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)   # (B, d_out)
        # Classifier
        out = {
            'action': self.h_action(pooled),
            'scope':  self.h_scope(pooled),
            'format': self.h_format(pooled),
            'spec':   self.h_spec(pooled),
            'target': self.h_target(pooled),
            'gate':   torch.sigmoid(self.gate(pooled).squeeze(-1)),
        }
        ps = self.ptr_s(h2).squeeze(-1)
        pe = self.ptr_e(h2).squeeze(-1)
        ps = ps.masked_fill(mask < 0.5, -1e9)
        pe = pe.masked_fill(mask < 0.5, -1e9)
        out['ptr_s'] = ps
        out['ptr_e'] = pe
        # Steering delta — из pooled, broadcast по T
        delta = self.W_steer(pooled).unsqueeze(1)         # (B, 1, 768)
        return out, delta


class FullSteeringGPT2(nn.Module):
    def __init__(self, adapter_layer=6, alpha=1.0):
        super().__init__()
        self.gpt = GPT2()
        self.adapter = FullSteeringAdapter()
        self.adapter_layer = adapter_layer
        self.alpha = alpha

    def freeze_gpt(self):
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.gpt.eval()

    def forward(self, input_ids, mask):
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)
        x = self.gpt.wte(input_ids) + self.gpt.wpe(pos)
        cls_out = None
        delta = None
        for i, layer in enumerate(self.gpt.layers):
            x = layer(x)
            if i == self.adapter_layer:
                cls_out, delta = self.adapter(x, mask)
                # Residual feedback (broadcast over T)
                x = x + self.alpha * delta
        x = self.gpt.ln_f(x)
        gpt_logits = x @ self.gpt.wte.weight.T
        return gpt_logits, cls_out, delta


def load_classifier_from_npz(model: FullSteeringGPT2, npz_path):
    """Inherit existing classifier weights, W_steer стартует в нуле."""
    w = dict(np.load(npz_path))
    sd = model.adapter.state_dict()
    sd['W1.weight'].copy_(torch.from_numpy(w['W1'].T.astype(np.float32)))
    sd['W1.bias'].copy_(torch.from_numpy(w['b1'].astype(np.float32)))
    sd['ln1.weight'].copy_(torch.from_numpy(w['ln1_g'].astype(np.float32)))
    sd['ln1.bias'].copy_(torch.from_numpy(w['ln1_b'].astype(np.float32)))
    sd['W2.weight'].copy_(torch.from_numpy(w['W2'].T.astype(np.float32)))
    sd['W2.bias'].copy_(torch.from_numpy(w['b2'].astype(np.float32)))
    sd['ln2.weight'].copy_(torch.from_numpy(w['ln2_g'].astype(np.float32)))
    sd['ln2.bias'].copy_(torch.from_numpy(w['ln2_b'].astype(np.float32)))
    head_map = [('action','h_action'),('scope','h_scope'),('format','h_format'),
                ('specificity','h_spec'),('target_kind','h_target')]
    for npz_name, t_name in head_map:
        sd[f'{t_name}.weight'].copy_(torch.from_numpy(w[f'h_{npz_name}_W'].T.astype(np.float32)))
        sd[f'{t_name}.bias'].copy_(torch.from_numpy(w[f'h_{npz_name}_b'].astype(np.float32)))
    sd['ptr_s.weight'].copy_(torch.from_numpy(w['ptr_s_W'].T.astype(np.float32)))
    sd['ptr_s.bias'].copy_(torch.zeros(1))
    sd['ptr_e.weight'].copy_(torch.from_numpy(w['ptr_e_W'].T.astype(np.float32)))
    sd['ptr_e.bias'].copy_(torch.zeros(1))
    sd['gate.weight'].copy_(torch.from_numpy(w['gate_W'].T.astype(np.float32)))
    sd['gate.bias'].copy_(torch.from_numpy(w['gate_b'].astype(np.float32)))
    # W_steer уже в нуле (init), не трогаем


# ============================================================
# Sanity: Pipeline = identity до обучения W_steer
# ============================================================
if __name__ == "__main__":
    print(f"Building FullSteeringGPT2 (resume torch_EN_BFCL)...")
    model = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
    load_gpt2_torch_weights(model.gpt)
    load_classifier_from_npz(model, str(Path(__file__).resolve().parent.parent / "weights" / "adapter_torch_EN_BFCL.npz"))
    model.freeze_gpt(); model.to(DEVICE); model.eval()

    print(f"  GPT-2: 124M frozen")
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable adapter (classifier + W_steer): {n_t:,}")

    # Тест 1: classifier работает (как раньше)
    prompt = "read src/auth.py"
    ids = encode(prompt)[:80]
    ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    mm = torch.ones((1, len(ids)), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits, cls, delta = model(ii, mm)
    print(f"\nClassifier (как раньше):")
    print(f"  '{prompt}' → action={ACTIONS[int(cls['action'].argmax(dim=1).item())]}, gate={float(cls['gate'].item()):.3f}")

    # Тест 2: vocab logits идентичны чистому GPT-2 (W_steer=0)
    print(f"\nW_steer=0 → должно быть identity (vocab logits = pure GPT-2):")
    delta_norm = float(delta.norm().item())
    print(f"  ||delta|| = {delta_norm:.6f}  (должно быть 0)")

    # Тест 3: один step обучения steering на gold token двигает GPT-2 logits
    print(f"\n=== Тест steering: prompt 'The capital of France is' → push to ' Paris' ===")
    prompt2 = "The capital of France is"
    ids2 = encode(prompt2)
    ii2 = torch.tensor([ids2], dtype=torch.long, device=DEVICE)
    mm2 = torch.ones((1, len(ids2)), dtype=torch.float32, device=DEVICE)
    paris = encode(" Paris")[0]

    with torch.no_grad():
        l0, _, _ = model(ii2, mm2)
        p0 = float(F.softmax(l0[0, -1], dim=0)[paris])
        top1_0 = decode([int(l0[0, -1].argmax().item())])
    print(f"  before: top1='{top1_0}', prob[' Paris']={p0:.4f}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for step in range(5):
        opt.zero_grad()
        l, _, _ = model(ii2, mm2)
        loss = F.cross_entropy(l[:, -1, :], torch.tensor([paris], device=DEVICE))
        loss.backward()
        opt.step()
        with torch.no_grad():
            ll, _, _ = model(ii2, mm2)
            p = float(F.softmax(ll[0, -1], dim=0)[paris])
            t = decode([int(ll[0, -1].argmax().item())])
            print(f"  step {step+1}: top1='{t}', prob[' Paris']={p:.4f}")

    # Также проверим что classifier не сломался после steering обучения
    with torch.no_grad():
        ll2, cls2, _ = model(ii, mm)
    print(f"\nClassifier ПОСЛЕ steering обучения (должен сохраниться):")
    print(f"  '{prompt}' → action={ACTIONS[int(cls2['action'].argmax(dim=1).item())]}, gate={float(cls2['gate'].item()):.3f}")
