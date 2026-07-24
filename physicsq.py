"""Differentiable photon-budget twin of a delocalized photonic inference link.

Models the Netcast-style setting (Sludds et al., Science 378, 270, 2022):
a central server broadcasts weight-encoded light; each edge client computes
matrix--vector products by modulating and photodetecting, with signed
weights carried on differential (push--pull) rails. At the photon-starved
operating points of interest (around one photon per MAC) the dominant
impairment is photodetection shot noise --- Poisson counting statistics,
the standard quantum limit (SQL) of direct detection --- with dark/
background counts, residual calibration (trim) error on the received
weights, and ADC quantization as the hardware impairments.

The twin uses the Gaussian (reparameterized) approximation of the Poisson
rail counts, which is differentiable and accurate to a few percent for
the count levels integrated per output (n_bar * N photons), so pathwise
gradients flow through the chain during training.

Photon accounting: n_bar photons per MAC on average across a layer; the
per-output signal variance follows the exact SQL of the differential
rails, Var[y_hat_i] = a_i * a_mean / (n_bar N) with a_i = sum_j |W_ij x_j|.
"""
import math
import torch

H_NU_1550 = 1.2816e-19        # photon energy at 1550 nm (J)
E_ADC = 1e-12                 # J per ADC sample (same constant as Ref. WISE)
E_DIG = 1e-12                 # J per digital op


class PhotonChain:
    """Stochastic, differentiable photonic MVM: y_hat = Q_b[ Wx + shot + dark ].

    Parameters
    ----------
    n_photon : mean photons per MAC (None = noiseless digital chain).
    dark_frac : dark/background counts as a fraction of the signal photon
        budget per output slot (rho_d).
    cal_sigma : std of residual multiplicative calibration (trim) error on
        the received weights, W * (1 + delta), delta ~ N(0, s^2).
    adc_bits : ADC resolution (None = ideal), +/-4 sigma clipping, STE.
    """

    def __init__(self, n_photon=None, dark_frac=0.0, cal_sigma=0.0,
                 adc_bits=None):
        self.n_photon = n_photon
        self.dark_frac = dark_frac
        self.cal_sigma = cal_sigma
        self.adc_bits = adc_bits

    def _quantize(self, t):
        b = self.adc_bits
        rng = 4.0 * t.detach().pow(2).mean().sqrt() + 1e-12
        step = 2 * rng / (2 ** b)
        q = torch.clamp(t, -rng, rng)
        q = torch.round(q / step) * step
        return t + (q - t).detach()

    def mvm(self, x, W):
        """x: (B, N) real, W: (M, N) real -> (B, M) real."""
        if self.cal_sigma and self.cal_sigma > 0:
            delta = torch.randn_like(W) * self.cal_sigma
            W = W * (1.0 + delta)
        y = x @ W.t()
        if self.n_photon is not None:
            N = x.shape[1]
            a = x.abs() @ W.abs().t()                 # rail intensity per output
            a_mean = a.detach().mean().clamp_min(1e-12)
            var_shot = a.detach() * a_mean / (self.n_photon * N)
            var_dark = 2.0 * self.dark_frac * a_mean ** 2 / (self.n_photon * N)
            sigma = torch.sqrt(var_shot + var_dark + 1e-30)
            y = y + sigma * torch.randn_like(y)
        if self.adc_bits is not None:
            y = self._quantize(y)
        return y


# ----------------------------- photon/energy accounting --------------------
def model_budget(n_photon, layer_dims, link_loss_db=30.0):
    """Photon budget and energy per inference for an FC stack.

    Returns the client-side received photon budget (photons/inference and
    its optical energy), the server-side laser energy through a link with
    `link_loss_db` loss, and the client electronic overheads (ADC + digital
    decode), which are photon-independent.
    """
    macs, samples, outputs = 0, 0, 0
    for (n, m) in layer_dims:
        macs += n * m
        samples += 2 * m                # two differential rails per output
        outputs += m
    phot = n_photon * macs
    e_opt = phot * H_NU_1550
    e_laser = e_opt * 10 ** (link_loss_db / 10.0)
    e_elec = samples * E_ADC + outputs * E_DIG
    return {"macs": macs, "photons": phot, "E_opt": e_opt,
            "E_laser": e_laser, "E_elec": e_elec,
            "E_client": e_opt + e_elec}


# ----------------------------- validation ----------------------------------
def validate_ip_rmse(n=4096, trials=400, nbar_grid=(0.25, 0.5, 1, 2, 4, 8, 16, 64),
                     seed=0):
    """Monte-Carlo IP error of the twin vs the analytic shot-noise (SQL) law.

    Random nonnegative intensities x ~ U[0,1], signed weights w ~ U[-1,1].
    Analytic SQL for this ensemble: Var[c_hat] = a * a_mean/(n_bar N) with
    a = sum |w_j x_j|; normalized RMSE_pred = sqrt(E[a] * a_mean) / sqrt(n_bar N)/sqrt(N).
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for nb in nbar_grid:
        errs, preds = [], []
        for _ in range(trials):
            x = torch.rand(1, n, generator=g)
            w = 2 * torch.rand(1, n, generator=g) - 1
            chain = PhotonChain(n_photon=nb)
            c_hat = chain.mvm(x, w)[0, 0]
            c = (x * w).sum()
            errs.append(((c_hat - c) ** 2 / n).item())
            a = (x.abs() * w.abs()).sum()
            preds.append((a * a / (nb * n) / n).item())   # a_mean ~= a (1 draw)
        out[str(nb)] = {"rmse": math.sqrt(sum(errs) / len(errs)),
                        "sql": math.sqrt(sum(preds) / len(preds))}
    return out
