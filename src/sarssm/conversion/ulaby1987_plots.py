"""
Additional plots for the Ulaby model (see ulaby1987.py).

Run with `python -m sarssm.conversion.ulaby1987_plots`.
Plots are saved to the `./sarssm_plots/ulaby1987_plots` folder,
in the current working directory, the folder is created if it does not exist.
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sarssm.conversion import ulaby1987


def plot_corn_dielectrics_ulaby():
    out_folder = pathlib.Path("./sarssm_plots/ulaby1987_plots")
    out_folder.mkdir(parents=True, exist_ok=True)
    m_g = np.linspace(0.05, 0.7, 100)
    for band, frequency_hz in [("X", 9.6e9), ("C", 5.3e9), ("L", 1.325e9)]:
        eps = ulaby1987.corn_moisture_to_eps_ulaby(m_g, frequency_hz)
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.set_title(f"Corn leaf dielectrics, Ulaby model\n{band}-band, {frequency_hz * 1e-9:.3f} GHz")
        ax.plot(m_g, eps.real, label="real")
        ax.plot(m_g, np.abs(eps.imag), label="|imag|")
        ax.set_xlabel("Gravimetric moisture")
        ax.set_ylabel("Dielectric constant")
        ax.set_xlim(0.05, 0.7)
        ax.set_ylim(0, 35)
        ax.legend()
        ax.grid()
        fig.savefig(out_folder / f"ulaby_corn_dielectrics_{band}_band.png", dpi=400)
        plt.close("all")


if __name__ == "__main__":
    plot_corn_dielectrics_ulaby()
