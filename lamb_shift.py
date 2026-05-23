"""
lamb_shift.py
-------------
First-order perturbation theory estimate of the Uehling contribution
to the hydrogen 2s Lamb shift.

Natural units throughout: hbar = c = m_e = 1.
Bohr radius a0 = 1/alpha ~ 137.036 (in units of 1/m_e).
"""

import numpy as np
from scipy.integrate import quad
from uehling import V_uehling, ALPHA

# Bohr radius in natural units (1/m_e)
A0 = 1 / ALPHA   # ~ 137.036


def psi_2s(r: float) -> float:
    """
    Hydrogen 2s radial wavefunction in natural units.
    R_20(r) such that the full wavefunction is psi = R_20(r) * Y_00.

    Normalized: integral |psi|^2 r^2 dr = 1
    """
    rho = r / A0
    norm = (1 / A0) ** (3/2) / (4 * np.sqrt(2))
    return norm * (2 - rho) * np.exp(-rho / 2)


def integrand_lamb(r: float) -> float:
    """
    Integrand for the energy shift:
    |psi_2s(r)|^2 * V_Uehling(r) * r^2
    """
    psi = psi_2s(r)
    return psi**2 * V_uehling(r) * r**2


def compute_uehling_lamb_shift() -> dict:
    """
    Compute the Uehling contribution to the hydrogen 2s energy shift.

    Returns a dict with the result in natural units, meV, and MHz.
    """
    # Integration range: Uehling potential is negligible beyond r ~ 0.1/m_e
    # But the wavefunction extends to r ~ a0 ~ 137, so we integrate fully
    # Use a split to handle the sharp Uehling feature near r=0
    result, error = quad(
        integrand_lamb,
        0.0,
        np.inf,
        limit=500,
        points=[0.001, 0.01, 0.1, 1.0],  # hint the integrator about structure
        epsabs=1e-20,
        epsrel=1e-8,
    )

    # Convert to physical units
    # result is in m_e (natural units)
    # 1 m_e = 0.51099895 MeV = 510998.95 meV
    me_to_meV = 510998.95   # meV per m_e
    # 1 meV = 241.799 MHz (via E = h*f)
    meV_to_MHz = 241.799

    result_meV = result * me_to_meV
    result_MHz = result_meV * meV_to_MHz

    return {
        "delta_E_natural": result,
        "delta_E_meV":     result_meV,
        "delta_E_MHz":     result_MHz,
        "integration_error": error,
    }


if __name__ == "__main__":
    print("Computing Uehling contribution to hydrogen 2s Lamb shift...")
    res = compute_uehling_lamb_shift()

    print(f"\nResult:")
    print(f"  ΔE (natural units) = {res['delta_E_natural']:.6e} m_e")
    print(f"  ΔE                 = {res['delta_E_meV']:.4f} meV")
    print(f"  ΔE                 = {res['delta_E_MHz']:.2f} MHz")
    print(f"  Integration error  = {res['integration_error']:.2e}")
    print(f"\nKnown value (Uehling contribution to 2s₁/₂ Lamb shift): ~ -27.2 MHz")
    print(f"Agreement: {abs(res['delta_E_MHz'] - (-27.2)) / 27.2 * 100:.1f}% deviation")