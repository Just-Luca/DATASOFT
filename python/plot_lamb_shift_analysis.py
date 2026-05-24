import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lamb_shift import psi_2s, integrand_lamb, A0
from uehling import V_uehling_v


def make_figure(output_path: str):
    # --------------------------------------------------
    # Radial coordinates
    # --------------------------------------------------
    r = np.linspace(1e-4, 20, 2000)
    r_short = np.logspace(-3, 1, 300)

    # Use constrained_layout instead of tight_layout
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.5),
        constrained_layout=True,
    )

    # --------------------------------------------------
    # Panel 1: 2s radial probability density
    # --------------------------------------------------
    try:
        # Faster if psi_2s supports NumPy arrays
        density = psi_2s(r) ** 2 * r**2
    except Exception:
        # Fallback if psi_2s only accepts scalars
        density = np.array([psi_2s(ri) ** 2 * ri**2 for ri in r])

    axes[0].plot(r / A0, density, lw=1.8)

    axes[0].set_xlabel(r"$r / a_0$")
    axes[0].set_ylabel(r"$|\psi_{2s}|^2 r^2$")
    axes[0].set_title("2s radial probability density")
    axes[0].grid(alpha=0.3)

    # --------------------------------------------------
    # Panel 2: Uehling potential
    # --------------------------------------------------
    try:
        # Faster if V_uehling_v supports NumPy arrays
        uehling_vals = V_uehling_v(r_short)
    except Exception:
        # Fallback if scalar-only
        uehling_vals = np.array([V_uehling_v(ri) for ri in r_short])

    axes[1].semilogx(
        r_short,
        uehling_vals * 1e6,
        lw=1.8,
    )

    axes[1].set_xlabel(r"$r$ [$1/m_e$]")

    # FIXED: use single backslashes in raw string
    axes[1].set_ylabel(
        r"$V_\mathrm{Uehling} \times 10^6$ [$m_e$]"
    )

    axes[1].set_title("Uehling potential (short range)")
    axes[1].grid(alpha=0.3)

    # --------------------------------------------------
    # Panel 3: Energy shift integrand
    # --------------------------------------------------
    try:
        # Faster if integrand_lamb supports NumPy arrays
        integrand = integrand_lamb(r)
    except Exception:
        # Fallback if scalar-only
        integrand = np.array([integrand_lamb(ri) for ri in r])

    axes[2].plot(r, integrand, lw=1.8)

    axes[2].set_xlabel(r"$r$ [$1/m_e$]")
    axes[2].set_ylabel("Integrand")
    axes[2].set_title("Energy shift integrand")
    axes[2].set_xlim([0, 2])
    axes[2].grid(alpha=0.3)

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Saved figure to: {output_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lamb shift analysis plots."
    )

    parser.add_argument(
        "--output",
        default="../figures/lamb_shift_analysis.png",
        help="Output image path.",
    )

    args = parser.parse_args()

    make_figure(args.output)


if __name__ == "__main__":
    main()