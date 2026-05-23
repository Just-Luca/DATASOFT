
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lamb_shift import psi_2s, integrand_lamb, A0
from uehling import V_uehling_v


def make_figure(output_path: str):
    # Radial coordinates
    r = np.linspace(1e-4, 20, 2000)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # --------------------------------------------------
    # Panel 1: 2s radial probability density
    # --------------------------------------------------
    density = [psi_2s(ri) ** 2 * ri**2 for ri in r]

    axes[0].plot(r / A0, density, lw=1.8)

    axes[0].set_xlabel(r"$r / a_0$")
    axes[0].set_ylabel(r"$|\psi_{2s}|^2 r^2$")
    axes[0].set_title("2s radial probability density")

    # --------------------------------------------------
    # Panel 2: Uehling potential
    # --------------------------------------------------
    r_short = np.logspace(-3, 1, 300)

    axes[1].semilogx(
        r_short,
        V_uehling_v(r_short) * 1e6,
        lw=1.8,
    )

    axes[1].set_xlabel(r"$r$ [$1/m_e$]")
    axes[1].set_ylabel(r"$V_\\mathrm{Uehling} \\times 10^6$ [$m_e$]")
    axes[1].set_title("Uehling potential (short range)")

    # --------------------------------------------------
    # Panel 3: Energy shift integrand
    # --------------------------------------------------
    integrand = [integrand_lamb(ri) for ri in r]

    axes[2].plot(r, integrand, lw=1.8)

    axes[2].set_xlabel(r"$r$ [$1/m_e$]")
    axes[2].set_ylabel("Integrand")
    axes[2].set_title("Energy shift integrand")
    axes[2].set_xlim([0, 2])

    # --------------------------------------------------
    # Final formatting + save
    # --------------------------------------------------
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=180, bbox_inches="tight")

    print(f"Saved figure to: {output_path}")


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