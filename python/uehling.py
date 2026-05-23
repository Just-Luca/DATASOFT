"""
uehling.py
----------
Numerical computation of the Uehling potential — the leading QED vacuum
polarization correction to the Coulomb potential.

Natural units: hbar = c = m_e = 1.
All lengths in units of the electron Compton wavelength 1/m_e.
All energies in units of m_e.

Reference: Uehling, Phys. Rev. 48, 55 (1935).
"""

import numpy as np
from scipy.integrate import quad

ALPHA = 1 / 137.036          # fine structure constant
ME = 1.0                     # electron mass in natural units


def _integrand(u: float, r: float) -> float:
    """
    Integrand of the Uehling integral.

    I(u, r) = exp(-2*r*u) * (1 + 1/(2u²)) * sqrt(u²-1) / u²

    Parameters
    ----------
    u : float
        Integration variable, u >= 1.
    r : float
        Radial distance in units of 1/m_e.
    """
    return np.exp(-2 * r * u) * (1 + 0.5 / u**2) * np.sqrt(u**2 - 1) / u**2


def uehling_integral(r: float) -> float:
    """
    Compute the dimensionless Uehling integral at distance r.

    ∫₁^∞ exp(-2ru) (1 + 1/2u²) √(u²-1)/u² du
    """
    result, _ = quad(
        _integrand,
        1.0,
        np.inf,
        args=(r,),
        limit=200,
        epsabs=1e-12,
        epsrel=1e-10,
    )
    return result


def V_coulomb(r: float) -> float:
    """Classical Coulomb potential: V = -alpha/r"""
    return -ALPHA / r


def V_uehling(r: float) -> float:
    """
    Uehling QED correction to the Coulomb potential.

    V_Uehling(r) = -(alpha/r) * (2*alpha / 3*pi) * integral(r)
    """
    prefactor = -ALPHA / r * (2 * ALPHA) / (3 * np.pi)
    return prefactor * uehling_integral(r)


def V_total(r: float) -> float:
    """Full QED-corrected potential (Coulomb + Uehling)."""
    return V_coulomb(r) + V_uehling(r)


# --- Vectorized versions for array inputs ---

V_coulomb_v = np.vectorize(V_coulomb)
V_uehling_v = np.vectorize(V_uehling)
V_total_v   = np.vectorize(V_total)


# --- Asymptotic approximations for validation ---

def V_uehling_short(r: float) -> float:
    """
    Short-range approximation (r << 1):
    V ~ -(alpha/r) * (2alpha/3pi) * (ln(1/r) - gamma_E + 5/6)
    """
    euler_gamma = 0.5772156649015329
    return -(ALPHA / r) * (2 * ALPHA / (3 * np.pi)) * (
        np.log(1 / r) - euler_gamma + 5 / 6
    )


def V_uehling_long(r: float) -> float:
    """
    Long-range approximation (r >> 1):
    V ~ -(alpha/r) * (alpha / 2*sqrt(pi)) * (3/2) * exp(-2r) / (2r)^(3/2)
    """
    return -(ALPHA / r) * (ALPHA / (2 * np.sqrt(np.pi))) * (
        1.5 * np.exp(-2 * r) / (2 * r) ** 1.5
    )