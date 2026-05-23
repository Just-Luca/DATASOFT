# QED Vacuum Polarization — Uehling Potential & Hydrogen Lamb Shift

A numerical study of the leading quantum electrodynamics (QED) correction to the Coulomb potential, implemented in both **Mathematica** and **Python**, with a first-order perturbative estimate of the Uehling contribution to the hydrogen 2s Lamb shift.

---

## A note on context

This project was built as a **learning exercise**, not as a demonstration of pre-existing expertise.

My background is in theoretical physics — quantum mechanics, the Standard Model, physics beyond the Standard Model — and in scientific computing with Python. I did not have prior hands-on experience with QED radiative corrections or the Uehling potential specifically before starting this. I also had not used Mathematica at a research level before.

What I did have was the theoretical foundation to pick these topics up quickly: I knew quantum field theory well enough to read and follow a one-loop calculation, I had worked with numerical integration and perturbation theory before, and I knew how to structure a scientific codebase.

I am sharing this because I think the ability to enter an unfamiliar area of physics, understand it at a working level, and produce a verified numerical result is more relevant to research and development work than a list of things I already knew.

---

## Physics background

### The Coulomb potential and its QED correction

In classical electrodynamics, the potential between two charges separated by distance $r$ is the Coulomb potential:

$$V_\text{Coulomb}(r) = -\frac{\alpha}{r}$$

(in natural units, $\hbar = c = m_e = 1$, with $\alpha \approx 1/137$ the fine structure constant).

In QED, the vacuum is not empty. Virtual electron-positron pairs are continuously created and annihilated, and a source charge polarizes this virtual medium. At one loop, this modifies the effective photon propagator and introduces a correction to the Coulomb potential at short distances.

### The vacuum polarization tensor

The correction is encoded in the one-loop photon self-energy (vacuum polarization tensor):

$$\Pi^{\mu\nu}(q) = (q^\mu q^\nu - g^{\mu\nu} q^2)\, \Pi(q^2)$$

After Pauli–Villars regularization (which removes the UV divergence by subtracting a fictitious heavy-particle contribution) and renormalization at $q^2 = 0$ (which fixes the photon to remain massless), the renormalized scalar function is:

$$\Pi(q^2) = \frac{\alpha}{\pi} \int_0^1 dz\; z(1-z)\, \ln\!\left(1 - \frac{q^2\, z(1-z)}{m_e^2}\right)$$

The cutoff mass drops out entirely after renormalization. The result is finite and regulator-independent.

### The Uehling potential

Fourier-transforming the corrected propagator back to position space yields the Uehling potential — the real-space form of the vacuum polarization correction:

$$V_\text{Uehling}(r) = -\frac{\alpha}{r} \cdot \frac{2\alpha}{3\pi} \int_1^\infty du\; e^{-2 m_e r\, u}\left(1 + \frac{1}{2u^2}\right)\frac{\sqrt{u^2-1}}{u^2}$$

where $u$ is a reparametrization of the Feynman parameter. Key features:

- The correction is **attractive** (same sign as Coulomb) and **short-range**: the $e^{-2m_e r u}$ factor confines it to distances $r \lesssim 1/(2m_e)$, the electron Compton wavelength (~0.002 Å).
- At **short range** ($r \ll 1/m_e$): the correction grows as $\sim \ln(1/r)$, strengthening the attraction logarithmically.
- At **long range** ($r \gg 1/m_e$): the correction falls off exponentially, recovering the classical Coulomb potential.
- The fractional correction $V_\text{Uehling}/V_\text{Coulomb}$ peaks at roughly 0.1% near $r \sim 0.01/m_e$.

### Application: the hydrogen Lamb shift

Because the Uehling potential is so short-range, it acts as a perturbation only on atomic states with non-zero probability density at the origin — that is, $s$-states ($\ell = 0$). The first-order energy shift of the hydrogen 2s state is:

$$\Delta E_{2s} = \langle 2s | V_\text{Uehling}(r) | 2s \rangle = \int_0^\infty |\psi_{2s}(r)|^2\, V_\text{Uehling}(r)\, r^2\, dr$$

This integral has no closed form and is evaluated numerically. The result is the Uehling contribution to the well-known **Lamb shift** — the experimentally observed splitting between the 2s$_{1/2}$ and 2p$_{1/2}$ levels of hydrogen (which are degenerate in the Dirac equation but split by QED effects).

---

## Implementation

### Mathematica (`mathematica/uehling_potential.nb`)

The Uehling integral is evaluated with `NIntegrate` using the `GaussKronrod` method. The notebook:

- Defines the integrand and the full potential as a function of $r$
- Verifies the numerical result against the known short-range ($r \ll 1$) and long-range ($r \gg 1$) asymptotic approximations at two test points
- Computes a log-spaced grid of $(r, V_\text{Uehling}(r))$ values
- Produces two plots: the full potential on a log-log scale, and the fractional correction $V_\text{Uehling}/V_\text{Coulomb}$ on a semilog scale
- Exports the numerical data to CSV for cross-validation in Python

The `?NumericQ` pattern guard on all functions that wrap `NIntegrate` is essential — without it, Mathematica attempts symbolic evaluation and fails.

### Python (`python/uehling.py`, `python/lamb_shift.py`)

The Python implementation uses `scipy.integrate.quad` to evaluate the same integral independently. The module is structured for reuse: pure functions with docstrings, vectorized wrappers for array inputs, and the asymptotic approximations as separate, testable functions.

The Lamb shift calculation (`lamb_shift.py`) evaluates the perturbation theory integral using `quad` with split-point hints near $r = 0$, where the Uehling potential is sharpest and the integrand structure requires guidance for the adaptive integrator.

Unit conversion chain:
- Natural units ($m_e = 1$) → meV: multiply by $m_e c^2 = 510{,}998.95\,\text{meV}$
- meV → MHz: multiply by $241.799\,\text{MHz/meV}$

### Cross-validation (`notebooks/validation.ipynb`)

The Python results are compared point-by-point against the Mathematica CSV export across the full $r$ grid.

| Metric | Value |
|---|---|
| Max relative error (Python vs. Mathematica) | < 1 × 10⁻⁶ |
| Mean relative error | < 1 × 10⁻⁸ |
| Uehling Lamb shift (this work) | −27.3 MHz |
| Known value (Uehling contribution, 2s₁/₂) | −27.2 MHz |
| Deviation | ~0.4% |

Agreement to better than one part in a million across both implementations, and within 0.4% of the known physical result, confirms that the numerics are correct.

---

## Repository structure

```
DATASOFT/
├── mathematica/
│   └── uehling_potential.nb          # Mathematica notebook
├── python/
│   ├── uehling.py                    # core module: V_Uehling(r)
│   ├── lamb_shift.py                 # perturbation theory: ΔE_2s
│   ├── validate_uehling.py           # cross-validation, Python vs. Mathematica
│   ├── plot.py
│   └── lamb_shift_analysis.py
├── data/
│   ├── uehling_data.csv              # exported from Mathematica
│   └── uehling_ratio.csv
├── figures/
│   ├── uehling_potential.png         # main result figure
│   └── lamb_shift_analysis.png       # wavefunction overlap analysis
└── README.md
```

---

## Figures

### QED-corrected potential

The two-panel figure (`figures/uehling_potential.png`) shows:

- **Top panel**: $|V_\text{Coulomb}(r)|$ and $|V_\text{Coulomb}(r) + V_\text{Uehling}(r)|$ on a log-log scale. The two lines are indistinguishable at large $r$ and diverge slightly at small $r$, where the QED correction is strongest.
- **Bottom panel**: the fractional correction $V_\text{Uehling}(r) / V_\text{Coulomb}(r)$ on a semilog scale. The correction peaks near $r \sim 0.01/m_e$ and falls off rapidly in both directions.

### Lamb shift analysis

The three-panel figure (`figures/lamb_shift_analysis.png`) shows the 2s radial probability density, the Uehling potential at short range, and the overlap integrand $|\psi_{2s}|^2 V_\text{Uehling}(r) r^2$ whose integral gives $\Delta E_{2s}$. The overlap integrand is sharply peaked near the origin, confirming that the energy shift is driven entirely by the short-range behavior of the potential.

---

## Requirements

### Python
```
numpy
scipy
matplotlib
jupyter
```
Install with: `pip install numpy scipy matplotlib jupyter`

### Mathematica
Wolfram Mathematica 12 or later. The notebook uses only built-in functions (`NIntegrate`, `ListLogLogPlot`, `Export`).

---

## References

1. Uehling, E. A. (1935). *Polarization effects in the positron theory*. Physical Review, 48(1), 55–63.
2. Peskin, M. E., & Schroeder, D. V. (1995). *An Introduction to Quantum Field Theory*. Addison-Wesley. §7.5.
3. Itzykson, C., & Zuber, J.-B. (1980). *Quantum Field Theory*. McGraw-Hill. §7.1.
4. Eides, M. I., Grotch, H., & Shelyuto, V. A. (2001). *Theory of light hydrogenlike atoms*. Physics Reports, 342(2-3), 63–261. (For the known Lamb shift value.)

---

## License

MIT
