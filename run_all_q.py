"""PILOT-Q master experiment driver (checkpoints results/results.json)."""
import json
import os
import sys
import time
import torch

from physicsq import validate_ip_rmse, model_budget
from data import DATASETS
from modelsq import RealFC
from trainq import train_model, accuracy
from experimentsq import (acc_vs_nbar, impairment_sweeps, closed_loop,
                          oracle_bound, NBAR_GRID, IMPAIRED)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
RESULTS_PATH = os.path.join(RES, "results.json")

EPOCHS = {"mnist": 25, "fashion": 25, "fsdd": 100}
# photon-budget training band per task (harder tasks need a gentler band)
BAND = {"mnist": (0.25, 8.0), "fashion": (0.5, 8.0), "fsdd": (0.25, 8.0)}
EPOCHS_PITL = {"mnist": 35, "fashion": 35, "fsdd": 140}
TAUS = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
PAIRS = [(0.25, 1), (0.5, 2), (1, 4), (2, 8), (0.5, 4), (4, 16)]


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(r):
    with open(RESULTS_PATH, "w") as f:
        json.dump(r, f, indent=1)


def stage_validate(r):
    print("== Twin validation vs analytic SQL ==", flush=True)
    r["ip_validation"] = validate_ip_rmse(n=4096, trials=400)
    r["ip_validation_n32768"] = validate_ip_rmse(n=32768, trials=60)
    save_results(r)


def run_dataset(r, name):
    cfg = DATASETS[name]
    print(f"== Dataset {name} ==", flush=True)
    (xtr, ytr), (xte, yte) = cfg["loader"]()
    Xte = torch.from_numpy(xte); Yte = torch.from_numpy(yte)
    dims = (cfg["in_dim"],) + cfg["hidden"] + (10,)
    rd = r.setdefault(name, {})
    rd["dims"] = list(dims)
    rd["n_train"], rd["n_test"] = len(xtr), len(xte)

    for mode in ["digital", "pitlq", "pitlq_full"]:
        ck = os.path.join(RES, f"{name}_{mode}.pt")
        if os.path.exists(ck):
            model = RealFC(dims); model.load_state_dict(torch.load(ck))
            print(f"  loaded checkpoint {name}_{mode}", flush=True)
        else:
            eps = EPOCHS[name] if mode == "digital" else EPOCHS_PITL[name]
            model = train_model(xtr, ytr, dims, mode, epochs=eps, seed=42,
                                band=BAND[name])
            torch.save(model.state_dict(), ck)
        md = rd.setdefault(mode, {})
        md["digital_acc"] = accuracy(model, Xte, Yte, None)
        print(f"  {mode}: digital acc {md['digital_acc']:.4f}", flush=True)
        md["acc_vs_nbar"] = acc_vs_nbar(model, Xte, Yte)
        md["acc_vs_nbar_impaired"] = acc_vs_nbar(model, Xte, Yte, **IMPAIRED)
        md["impair_2ph"] = impairment_sweeps(model, Xte, Yte, n_photon=2.0)
        save_results(r)

    ldims = list(zip(dims[:-1], dims[1:]))
    for mode in ["pitlq", "digital"]:
        model = RealFC(dims)
        model.load_state_dict(torch.load(os.path.join(RES, f"{name}_{mode}.pt")))
        cl = rd.setdefault(f"closed_loop_{mode}", {})
        for (lo, hi) in PAIRS:
            cl[f"{lo}_{hi}"] = closed_loop(model, Xte, Yte, lo, hi, TAUS,
                                           dims=ldims)
            save_results(r)
        print(f"  closed loop ({mode}) done", flush=True)
    model = RealFC(dims)
    model.load_state_dict(torch.load(os.path.join(RES, f"{name}_pitlq.pt")))
    rd["closed_loop_impaired"] = closed_loop(model, Xte, Yte, 1, 4, TAUS,
                                             dims=ldims, **IMPAIRED)
    # oracle escalation bound (PILOT-Q model, clean chain)
    model = RealFC(dims)
    model.load_state_dict(torch.load(os.path.join(RES, f"{name}_pitlq.pt")))
    rd["oracle"] = {f"{lo}_{hi}": oracle_bound(model, Xte, Yte, lo, hi,
                                               dims=ldims)
                    for (lo, hi) in PAIRS}
    save_results(r)
    print(f"  {name} done", flush=True)


def stage_ablation(r):
    print("== Ablation: fixed photon budget (MNIST) ==", flush=True)
    (xtr, ytr), (xte, yte) = DATASETS["mnist"]["loader"]()
    Xte = torch.from_numpy(xte); Yte = torch.from_numpy(yte)
    dims = (784, 300, 100, 10)
    ck = os.path.join(RES, "mnist_pitlq_fixed1.pt")
    if os.path.exists(ck):
        model = RealFC(dims); model.load_state_dict(torch.load(ck))
    else:
        model = train_model(xtr, ytr, dims, "pitlq_fixed1",
                            epochs=EPOCHS_PITL["mnist"], seed=42)
        torch.save(model.state_dict(), ck)
    ab = r.setdefault("mnist", {}).setdefault("pitlq_fixed1", {})
    ab["digital_acc"] = accuracy(model, Xte, Yte, None)
    ab["acc_vs_nbar"] = acc_vs_nbar(model, Xte, Yte)
    save_results(r)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    r = load_results()
    t0 = time.time()
    if stage in ("validate", "all"):
        stage_validate(r)
    for name in ["mnist", "fashion", "fsdd"]:
        if stage in (name, "all"):
            run_dataset(r, name)
    if stage in ("ablation", "all"):
        stage_ablation(r)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)
