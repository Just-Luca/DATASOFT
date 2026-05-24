"""
lamb_shift.py
-------------
First-order perturbation theory estimate of the Uehling contribution
to the hydrogen 2s Lamb shift.

Natural units throughout:
    ħ = c = m_e = 1

Bohr radius:
    a0 = 1 / alpha ≈ 137.036  (in units of 1/m_e)
"""

import numpy as np
from scipy.integrate import quad

from uehling import V_uehling, ALPHA

# --------------------------------------------------
# Constants
# --------------------------------------------------

# Bohr radius in natural units (1 / m_e)
A0 = 1.0 / ALPHA


def psi_2s(r: float) -> float:
    """
    Full hydrogen 2s spatial wavefunction ψ_200(r).

    Includes the Y_00 angular normalization.

    Normalization:
        ∫ |ψ|² d³r = 1
    """

    rho = r / A0

    # Full 3D normalized 2s wavefunction
    norm = (1.0 / A0) ** (3.0 / 2.0) / (
        4.0 * np.sqrt(2.0 * np.pi)
    )

    return norm * (2.0 - rho) * np.exp(-rho / 2.0)


def integrand_lamb(r: float) -> float:
    """
    Radial integrand for the Uehling energy correction.

    Since ψ already includes Y_00 normalization,
    the angular integration contributes a factor 4π.
    """

    psi = psi_2s(r)

    return 4.0 * np.pi * psi**2 * V_uehling(r) * r**2
    

def compute_uehling_lamb_shift() -> dict:
    """
    Compute the Uehling contribution to the hydrogen 2s Lamb shift.

    Returns
    -------
    dict
        Energy shift in:
            - natural units (m_e)
            - meV
            - MHz
    """

    # --------------------------------------------------
    # Split integration region
    # --------------------------------------------------
    #
    # Near r = 0:
    #     Uehling potential varies rapidly
    #
    # Large r:
    #     exponentially suppressed
    #
    # SciPy does not allow:
    #     np.inf together with points=[...]
    #
    # therefore split integral manually.
    # --------------------------------------------------

    # Short-distance region
    result1, err1 = quad(
        integrand_lamb,
        0.0,
        1.0,
        points=[1e-4, 1e-3, 1e-2, 1e-1],
        limit=500,
        epsabs=1e-20,
        epsrel=1e-8,
    )

    # Long-distance tail
    result2, err2 = quad(
        integrand_lamb,
        1.0,
        np.inf,
        limit=500,
        epsabs=1e-20,
        epsrel=1e-8,
    )

    result = result1 + result2
    error = err1 + err2

    # --------------------------------------------------
    # Unit conversions
    # --------------------------------------------------

    # 1 electron mass = 0.51099895 MeV
    #
    # Convert:
    #   MeV → eV : ×10^6
    #   eV  → meV: ×10^3
    #
    # Therefore:
    #   1 m_e = 5.1099895 × 10^8 meV
    #
    me_to_meV = 5.1099895e8

    # Energy-frequency conversion:
    #
    #   1 meV = 241.799 GHz
    #
    # Therefore:
    #   1 meV = 241799 MHz
    #
    meV_to_MHz = 241799.0

    result_meV = result * me_to_meV
    result_MHz = result_meV * meV_to_MHz

    return {
        "delta_E_natural": result,
        "delta_E_meV": result_meV,
        "delta_E_MHz": result_MHz,
        "integration_error": error,
    }


def main():

    print("Computing Uehling contribution to hydrogen 2s Lamb shift...")

    res = compute_uehling_lamb_shift()

    print("\nResult:")
    print(f"  ΔE (natural units) = {res['delta_E_natural']:.6e} m_e")
    print(f"  ΔE                 = {res['delta_E_meV']:.6e} meV")
    print(f"  ΔE                 = {res['delta_E_MHz']:.3f} MHz")
    print(f"  Integration error  = {res['integration_error']:.2e}")

    known_value = -27.2

    deviation = (
        abs(res["delta_E_MHz"] - known_value)
        / abs(known_value)
        * 100.0
    )

    print("\nReference value:")
    print("  Uehling contribution to hydrogen 2s Lamb shift ≈ -27.2 MHz")

    print(f"Deviation from reference: {deviation:.2f}%")


if __name__ == "__main__":
    main()