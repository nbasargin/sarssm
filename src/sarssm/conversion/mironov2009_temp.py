"""
Mironov et al. (2009) clay-based GRMDM in PyTorch (differentiable, broadcasting).

Coefficients: Mironov, Kosolapova & Fomin (2009), IEEE TGRS 47(7):2059-2070,
Using Eqs. (11)-(16) and (17)-(25) from the paper.
The erratum (p. 2085) only concerns Eq. (9) (Dobson SMDM conductivity) and does not affect this model.
Valid range of training data: 45 MHz - 26.5 GHz, 20-22 degC, clay 0-76 %.
"""

import math

import torch

EPS_VAC = 8.854e-12
EPS_INF = 4.9


def _debye_nk(f, eps0, tau, sigma):
    """Debye + conductivity (Mironov 2004 eq. 11) -> refractive index n and NAC kappa (eq. 7)."""
    wt = 2 * math.pi * f * tau
    er = EPS_INF + (eps0 - EPS_INF) / (1 + wt**2)
    ei = (eps0 - EPS_INF) / (1 + wt**2) * wt + sigma / (2 * math.pi * EPS_VAC * f)
    mag = torch.sqrt(er**2 + ei**2)
    return torch.sqrt((mag + er) / 2), torch.sqrt((mag - er) / 2)


def moisture_to_eps_mironov_temp(mv, clay, frequency):
    """
    IMPLEMENTATION NOT CHECKED!

    mv        : volumetric water content [m3/m3] (tensor, any shape)
    clay      : clay fraction 0..1 (tensor or float, broadcastable to mv)
    frequency : Hz (float or tensor, broadcastable)
    returns   : complex tensor eps' - 1j*eps''
    """
    mv = torch.as_tensor(mv)
    C = torch.as_tensor(clay, dtype=mv.dtype) * 100.0  # 2009 regressions use percent
    f = torch.as_tensor(frequency, dtype=mv.dtype)

    nd = 1.634 - 0.539e-2 * C + 0.2748e-4 * C**2
    kd = 0.03952 - 0.04038e-2 * C
    mvt = 0.02863 + 0.30673e-2 * C
    nb, kb = _debye_nk(f, 79.8 - 85.4e-2 * C + 32.7e-4 * C**2, 1.062e-11 + 3.450e-12 * 1e-2 * C, 0.3112 + 0.467e-2 * C)
    nu, ku = _debye_nk(f, torch.full_like(C, 100.0), torch.full_like(C, 8.5e-12), 0.3631 + 1.217e-2 * C)

    mvb = torch.minimum(mv, mvt)  # bound part
    mvu = torch.clamp(mv - mvt, min=0.0)  # free part
    n = nd + (nb - 1) * mvb + (nu - 1) * mvu
    k = kd + kb * mvb + ku * mvu
    return torch.complex(n**2 - k**2, -2 * n * k)
