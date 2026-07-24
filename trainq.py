"""Training protocols for PILOT-Q: conventional vs photon-budget-aware."""
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from physicsq import PhotonChain
from modelsq import RealFC

torch.set_num_threads(2)
P_CLEAN = 0.35
GAMMA = 8.0     # normalized-logit temperature (digital post-processing)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def logu(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_chain(mode, rng, band=(0.25, 8.0)):
    if mode == "digital":
        return None
    if rng.rand() < P_CLEAN:
        return None
    if mode == "pitlq":                     # shot-noise-aware
        return PhotonChain(n_photon=logu(rng, *band))
    if mode == "pitlq_full":                # full hardware-aware
        bits = rng.choice([4, 6, 8, None])
        return PhotonChain(n_photon=logu(rng, *band),
                           dark_frac=rng.uniform(0.0, 0.20),
                           cal_sigma=rng.uniform(0.0, 0.15),
                           adc_bits=bits)
    if mode == "pitlq_fixed1":              # ablation: single photon budget
        return PhotonChain(n_photon=1.0)
    raise ValueError(mode)


def train_model(xtr, ytr, dims, mode, epochs, batch=256, lr=1e-3, seed=42,
                log_every=5, band=(0.25, 8.0)):
    """Clean warmup 15% of epochs; lr x0.3 at 70%; clean anchor P_CLEAN."""
    set_seed(seed)
    rng = np.random.RandomState(seed + 1)
    model = RealFC(dims)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(xtr); Y = torch.from_numpy(ytr)
    n = len(X)
    warmup = max(1, int(0.15 * epochs)) if mode != "digital" else 0
    for ep in range(epochs):
        if ep == int(0.7 * epochs):
            for g in opt.param_groups:
                g["lr"] = lr * 0.3
        perm = torch.randperm(n)
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = X[idx], Y[idx]
            chain = None if ep < warmup else sample_chain(mode, rng, band)
            s = model(xb, chain)
            logits = GAMMA * s / (s.norm(dim=1, keepdim=True) + 1e-9)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb); tot += len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"    [{mode}] epoch {ep+1}/{epochs} "
                  f"loss {loss_sum/tot:.4f} acc {correct/tot:.4f}", flush=True)
    return model


@torch.no_grad()
def accuracy(model, X, Y, chain=None, batch=2000):
    correct = 0
    for i in range(0, len(X), batch):
        s = model(X[i:i + batch], chain)
        correct += (s.argmax(1) == Y[i:i + batch]).sum().item()
    return correct / len(X)
