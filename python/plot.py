"""
plot.py
-------
Generate the main figure: QED-corrected vs Coulomb potential,
plus fractional correction panel.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from uehling import V_coulomb_v, V_uehling_v, V_total_v

# Log-spaced r grid
r = np.logspace(-2, 1, 300)

V_c  = np.abs(V_coulomb_v(r))
V_t  = np.abs(V_total_v(r))
frac = V_uehling_v(r) / V_coulomb_v(r)   # fractional correction (positive: same sign)

# --- Figure setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
fig.subplots_adjust(hspace=0.08)

PURPLE = "#534AB7"
GRAY   = "#888780"

# Panel 1: log-log potential
ax1.loglog(r, V_c, color=GRAY,   lw=1.5, ls="--", label="Coulomb $|V|$")
ax1.loglog(r, V_t, color=PURPLE, lw=2.0,           label="Coulomb + Uehling $|V|$")
ax1.set_ylabel(r"$|V(r)|$ [$m_e$]", fontsize=11)
ax1.legend(fontsize=10, framealpha=0)
ax1.set_title("QED vacuum polarization correction to the Coulomb potential", fontsize=11)
ax1.grid(True, which="both", lw=0.4, alpha=0.4)

# Panel 2: fractional correction (semilog-x)
ax2.semilogx(r, frac, color=PURPLE, lw=2.0)
ax2.axhline(0, color=GRAY, lw=0.8, ls="--")
ax2.set_xlabel(r"$r$ [$1/m_e$]", fontsize=11)
ax2.set_ylabel(r"$V_\mathrm{Uehling} / V_\mathrm{Coulomb}$", fontsize=11)
ax2.grid(True, which="both", lw=0.4, alpha=0.4)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

plt.savefig("../figures/uehling_potential.png", dpi=180, bbox_inches="tight")
plt.show()
print("Figure saved.")