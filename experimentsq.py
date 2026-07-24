"""Evaluation protocols for PILOT-Q: photon sweeps, impairments, closed loop."""
import numpy as np
import torch
from physicsq import PhotonChain, model_budget
from trainq import accuracy

NBAR_GRID = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 64]
EVAL_SEEDS = [101, 202, 303, 404, 505]
IMPAIRED = dict(dark_frac=0.20, cal_sigma=0.15, adc_bits=4)


@torch.no_grad()
def acc_vs_nbar(model, X, Y, grid=NBAR_GRID, seeds=EVAL_SEEDS, **imp):
    out = {}
    for nb in grid:
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            accs.append(accuracy(model, X, Y, PhotonChain(n_photon=nb, **imp)))
        out[str(nb)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    return out


@torch.no_grad()
def impairment_sweeps(model, X, Y, n_photon=2.0, seeds=EVAL_SEEDS):
    res = {"dark": {}, "cal": {}, "adc": {}}
    for d in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            accs.append(accuracy(model, X, Y,
                        PhotonChain(n_photon=n_photon, dark_frac=d)))
        res["dark"][str(d)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    for c in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            accs.append(accuracy(model, X, Y,
                        PhotonChain(n_photon=n_photon, cal_sigma=c)))
        res["cal"][str(c)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    for b in [2, 3, 4, 5, 6, 8]:
        accs = []
        for s in seeds:
            torch.manual_seed(s)
            accs.append(accuracy(model, X, Y,
                        PhotonChain(n_photon=n_photon, adc_bits=b)))
        res["adc"][str(b)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
    return res


@torch.no_grad()
def _logits(model, X, chain, batch=2000):
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch], chain))
    return torch.cat(out)


@torch.no_grad()
def closed_loop(model, X, Y, nb_lo, nb_hi, taus, seeds=EVAL_SEEDS,
                dims=None, **imp):
    """Confidence-gated photon-budget escalation with pooled-count combining.

    Escalated inputs are re-measured at nb_hi and the two estimates are
    combined with photon-number weights (the maximum-likelihood pooling of
    the two exposures): y = (nb_lo*y1 + nb_hi*y2)/(nb_lo+nb_hi).
    Confidence = softmax margin of the low-budget logits.
    """
    B_lo = model_budget(nb_lo, dims)
    B_hi = model_budget(nb_hi, dims)
    acc = {t: [] for t in taus}
    frac = {t: [] for t in taus}
    for s in seeds:
        torch.manual_seed(s)
        l1 = _logits(model, X, PhotonChain(n_photon=nb_lo, **imp))
        l2 = _logits(model, X, PhotonChain(n_photon=nb_hi, **imp))
        p = torch.softmax(l1, dim=1)
        top2 = torch.topk(p, 2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        pred1 = l1.argmax(1)
        pred2 = ((nb_lo * l1 + nb_hi * l2) / (nb_lo + nb_hi)).argmax(1)
        for t in taus:
            esc = margin < t
            pred = torch.where(esc, pred2, pred1)
            acc[t].append((pred == Y).float().mean().item())
            frac[t].append(esc.float().mean().item())
    res = []
    for t in taus:
        f = float(np.mean(frac[t]))
        res.append({"tau": t, "acc_mean": float(np.mean(acc[t])),
                    "acc_std": float(np.std(acc[t])), "esc_frac": f,
                    "photons": B_lo["photons"] + f * B_hi["photons"],
                    "E_client": B_lo["E_client"] + f * B_hi["E_client"]})
    return {"nb_lo": nb_lo, "nb_hi": nb_hi, "points": res}


@torch.no_grad()
def oracle_bound(model, X, Y, nb_lo, nb_hi, seeds=EVAL_SEEDS, dims=None):
    """Upper bound on selective escalation: an oracle escalates exactly the
    inputs mispredicted at the low budget. Returns achievable (accuracy,
    photon budget) of the oracle policy."""
    from physicsq import model_budget
    B_lo = model_budget(nb_lo, dims)
    B_hi = model_budget(nb_hi, dims)
    accs, fracs = [], []
    for s in seeds:
        torch.manual_seed(s)
        l1 = _logits(model, X, PhotonChain(n_photon=nb_lo))
        l2 = _logits(model, X, PhotonChain(n_photon=nb_hi))
        pred1 = l1.argmax(1)
        esc = pred1 != Y
        pred2 = ((nb_lo * l1 + nb_hi * l2) / (nb_lo + nb_hi)).argmax(1)
        pred = torch.where(esc, pred2, pred1)
        accs.append((pred == Y).float().mean().item())
        fracs.append(esc.float().mean().item())
    f = float(np.mean(fracs))
    return {"nb_lo": nb_lo, "nb_hi": nb_hi, "acc": float(np.mean(accs)),
            "esc_frac": f, "photons": B_lo["photons"] + f * B_hi["photons"]}
