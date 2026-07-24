"""Open-dataset loaders.

MNIST      : LeCun et al., CC BY-SA 3.0, IDX files (GitHub mirror of the
             canonical distribution).
Fashion    : Xiao et al. 2017, MIT license, official Zalando repository.
FSDD       : Free Spoken Digit Dataset v1.0.10 (Jackson et al.),
             CC BY-SA 4.0, official repository. 3,000 recordings of
             spoken digits 0-9, six speakers, 8 kHz.

Audio front-end mirrors the WISE AudioMNIST pipeline: middle 0.5 s
(4,000 samples), STFT with 20-sample windows -> 200 frames x 20 bins,
amplitude-only spectrogram flattened to a 4,000-dim vector.
"""
import gzip
import os
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[:4], "big")
    if magic == 2051:
        n, r, c = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(3)]
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, r * c).astype(np.float32) / 255.0
    n = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)


def load_mnist(name="mnist"):
    d = os.path.join(ROOT, "mnist_repo" if name == "mnist" else "fmnist_repo/data/fashion")
    xtr = _load_idx(os.path.join(d, "train-images-idx3-ubyte.gz"))
    ytr = _load_idx(os.path.join(d, "train-labels-idx1-ubyte.gz"))
    xte = _load_idx(os.path.join(d, "t10k-images-idx3-ubyte.gz"))
    yte = _load_idx(os.path.join(d, "t10k-labels-idx1-ubyte.gz"))
    return (xtr, ytr), (xte, yte)


def _spectrogram(wav, n_samples=4000, win=20):
    """Amplitude STFT feature (WISE AudioMNIST recipe), 4,000-dim."""
    x = np.asarray(wav, dtype=np.float32)
    if len(x) < n_samples:                       # center-pad
        pad = n_samples - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    else:                                        # middle 0.5 s
        s = (len(x) - n_samples) // 2
        x = x[s:s + n_samples]
    frames = x.reshape(n_samples // win, win)    # 200 x 20
    spec = np.abs(np.fft.fft(frames, axis=1))    # amplitude only (drop phase)
    feat = spec.reshape(-1)
    return feat / (feat.max() + 1e-9)


def load_fsdd():
    """Standard FSDD split: recording indices 0-4 = test, 5-49 = train."""
    import soundfile as sf
    rec = os.path.join(ROOT, "fsdd", "recordings")
    xtr, ytr, xte, yte = [], [], [], []
    for fn in sorted(os.listdir(rec)):
        if not fn.endswith(".wav"):
            continue
        digit, _spk, idx = fn[:-4].split("_")
        wav, _sr = sf.read(os.path.join(rec, fn))
        feat = _spectrogram(wav)
        if int(idx) <= 4:
            xte.append(feat); yte.append(int(digit))
        else:
            xtr.append(feat); ytr.append(int(digit))
    return ((np.stack(xtr), np.array(ytr, np.int64)),
            (np.stack(xte), np.array(yte, np.int64)))


DATASETS = {
    "mnist":   dict(loader=lambda: load_mnist("mnist"),   in_dim=784,  hidden=(300, 100)),
    "fashion": dict(loader=lambda: load_mnist("fashion"), in_dim=784,  hidden=(300, 100)),
    "fsdd":    dict(loader=load_fsdd,                     in_dim=4000, hidden=(300, 100)),
}
