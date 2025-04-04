"""
Additional plots for the X-Bragg model (see hajnsek2003.py).

Run with `python -m sarssm.polsar.hajnsek2003_plots`.
Plots are saved to the `./sarssm_plots/hajnsek2003_plots` folder,
in the current working directory, the folder is created if it does not exist.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import sarssm
from sarssm.polsar import hajnsek2003


def plot_points(entropy_coords, alpha_mean_coords, eps_vals, title, save_to, cbar_label="eps"):
    plt.figure()
    plt.title(title)
    plt.scatter(
        entropy_coords, np.degrees(alpha_mean_coords), c=eps_vals, s=4, label="eps", vmin=0, vmax=40, cmap="viridis_r"
    )
    plt.xlabel("Entropy")
    plt.ylabel("Mean Alpha [degrees]")
    plt.xlim(0, 0.8)
    plt.ylim(0, 35)
    plt.colorbar(label=cbar_label)
    plt.grid()
    plt.savefig(save_to, dpi=200)
    plt.close("all")


def plot_grid(eps_lut, entropy_range, alpha_range, title, save_to, cbar_label="eps"):
    plt.figure()
    plt.title(title)
    plt.imshow(
        eps_lut.T,
        origin="lower",
        extent=(*entropy_range, *np.degrees(alpha_range)),
        aspect="auto",
        vmin=0,
        vmax=40,
        cmap="viridis_r",
    )
    plt.xlabel("Entropy")
    plt.ylabel("Mean Alpha [degrees]")
    plt.xlim(0, 0.8)
    plt.ylim(0, 35)
    plt.colorbar(label=cbar_label)
    plt.grid()
    plt.savefig(save_to, dpi=200)
    plt.close("all")


def plot_parameter_lookup_tables():
    print("Hajnsek 2003 - plot parameter lookup tables")
    out_folder = pathlib.Path("./sarssm_plots/hajnsek2003_plots")
    out_folder.mkdir(parents=True, exist_ok=True)
    theta_bounds = (np.radians(10.0), np.radians(60.0))
    eps_bounds = (2.0, 40.0)
    delta_bounds = (np.radians(0.0), np.radians(90.0))
    theta_grid, eps_grid, delta_grid = hajnsek2003._get_xbragg_parameter_grids(theta_bounds, eps_bounds, delta_bounds)
    t = hajnsek2003.xbragg_model(theta_grid, eps_grid, delta_grid)
    entropy, anisotropy, alpha_mean, alpha_dominant = sarssm.h_a_alpha_decomposition(t)
    for incidence_slice_index in range(0, 51, 10):
        selected_incidence_deg = np.rint(np.degrees(theta_grid[incidence_slice_index, 0, 0])).astype(np.int32)
        entropy_coords = entropy[incidence_slice_index].flatten()
        alpha_mean_coords = alpha_mean[incidence_slice_index].flatten()
        eps_vals = eps_grid[incidence_slice_index].flatten()
        entropy_range = (0.0, 0.8)
        alpha_range = (0.0, np.radians(35.0))
        eps_lut = hajnsek2003._entropy_alpha_param_lut(
            entropy_coords, alpha_mean_coords, eps_vals, entropy_range=entropy_range, alpha_range=alpha_range, steps=256
        )
        title = f"Eps in the entropy/alpha plane\nincidence = {selected_incidence_deg}°"
        plot_points(
            entropy_coords,
            alpha_mean_coords,
            eps_vals,
            title,
            save_to=out_folder / f"eps_points_incidence_{selected_incidence_deg}_deg.png",
            cbar_label="eps",
        )
        plot_grid(
            eps_lut,
            entropy_range,
            alpha_range,
            title,
            save_to=out_folder / f"eps_grid_incidence_{selected_incidence_deg}_deg.png",
            cbar_label="eps",
        )
        # soil moisture
        sm_vals = sarssm.eps_to_moisture_topp(eps_vals) * 100
        sm_lut = sarssm.eps_to_moisture_topp(eps_lut) * 100
        title = f"Moisture in the entropy/alpha plane\nincidence = {selected_incidence_deg}°"
        plot_points(
            entropy_coords,
            alpha_mean_coords,
            sm_vals,
            title,
            save_to=out_folder / f"sm_points_incidence_{selected_incidence_deg}_deg.png",
            cbar_label="Soil Moisture in %",
        )
        plot_grid(
            sm_lut,
            entropy_range,
            alpha_range,
            title,
            save_to=out_folder / f"sm_grid_incidence_{selected_incidence_deg}_deg.png",
            cbar_label="Soil Moisture in %",
        )


if __name__ == "__main__":
    plot_parameter_lookup_tables()
