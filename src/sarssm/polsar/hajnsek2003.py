"""
X-Bragg model for soil moisture estimation from fully-polarimetric PolSAR data.

I. Hajnsek, E. Pottier and S. R. Cloude
Inversion of surface parameters from polarimetric SAR
IEEE Transactions on Geoscience and Remote Sensing, 2003
DOI: 10.1109/TGRS.2003.810702
"""

import numpy as np
import scipy
import sarssm

# Forward model


def xbragg_model(theta, eps, delta):
    """
    Create polarimetric coherency matrices T (Pauli basis) from parameters according to the X-Bragg model.
    Input parameters are numpy arrays with the same shape (*B,).
    Parameters:
        theta - incidence angle, in radians
        eps - surface dielectrics
        delta - surface roughness distribution width, in radians
    Returns:
        X-Bragg T matrices, numpy array with shape (*B, 3, 3)
    """
    cos_t = np.cos(theta)
    sin2_t = np.sin(theta) ** 2
    sqrt = np.sqrt(eps - sin2_t)
    r_s = (cos_t - sqrt) / (cos_t + sqrt)
    r_p = (eps - 1) * (sin2_t - eps * (1 + sin2_t)) / ((eps * cos_t + sqrt) ** 2)
    c1 = np.abs(r_s + r_p) ** 2
    c2 = (r_s + r_p) * np.conj(r_s - r_p)
    c3 = 0.5 * np.abs(r_s - r_p) ** 2
    sinc2 = np.sinc(2 * delta / np.pi)
    sinc4 = np.sinc(4 * delta / np.pi)
    zeros = np.zeros_like(theta)
    t = np.array([[c1, c2 * sinc2, zeros], [c2 * sinc2, c3 * (1 + sinc4), zeros], [zeros, zeros, c3 * (1 - sinc4)]])
    return np.moveaxis(t, [0, 1], [-2, -1])  # move first two axes (correspond to the matrix) to the end


# Model inversion with lookup tables


def _get_xbragg_parameter_grids(theta_bounds, eps_bounds, delta_bounds):
    """
    Create a grid with parameters sampling the input space of the X-Bragg model.
    """
    theta_min, theta_max = theta_bounds
    eps_min, eps_max = eps_bounds
    delta_min, delta_max = delta_bounds
    theta_steps = int(np.rint(np.degrees(theta_max - theta_min) + 1))
    theta = np.linspace(theta_min, theta_max, theta_steps, endpoint=True)
    eps_steps = int(eps_max - eps_min + 1)
    eps = np.exp(np.linspace(np.log(eps_min), np.log(eps_max), eps_steps, endpoint=True))
    delta_steps = int(np.rint((np.degrees(delta_max - delta_min) + 1) / 2))
    delta = np.linspace(delta_min, delta_max, delta_steps, endpoint=True)
    theta_grid, eps_grid, delta_grid = np.meshgrid(theta, eps, delta, indexing="ij")
    return theta_grid, eps_grid, delta_grid


def _entropy_alpha_param_lut(
    entropy_coords, alpha_mean_coords, values, entropy_range=(0.0, 0.5), alpha_range=(0.0, np.radians(25.0)), steps=200
):
    """
    Interpolate values assigned to points in the entropy-alpha space.
    This effectively creates a lookup table where a parameter value can be retrieved from an entropy-alpha pair.
    """
    h_min, h_max = entropy_range
    a_min, a_max = alpha_range
    grid_x, grid_y = np.mgrid[h_min : h_max : steps * 1j, a_min : a_max : steps * 1j]
    value_coords = np.array([entropy_coords, alpha_mean_coords]).transpose((1, 0))
    grid = scipy.interpolate.griddata(value_coords, values, (grid_x, grid_y), method="linear")
    return grid


def _values_to_lut_indices(values, value_range, lut_slices):
    """
    Convert values to lookup table indices based on the value range and the number of lookup table slices.
    For values that would create invalid indices (e.g. values outside the value range),
    the index is set to 0 and the corresponding element `index_is_invalid` is set to True.
    """
    value_min, value_max = value_range
    step = (value_max - value_min) / (lut_slices - 1)
    indices_float = np.rint((values - value_min) / step)
    indices_float[np.isnan(indices_float)] = -1  # use negative value for invalid, nan cannot be cast to an int
    indices = indices_float.astype(np.int32)
    index_is_invalid = np.logical_or(indices < 0, indices >= lut_slices)
    indices[index_is_invalid] = 0
    return indices, index_is_invalid


def _create_xbragg_eps_lut():
    """
    Generate a lookup table that maps incidence_angle, entropy, and alpha triplets to the corresponding eps values.
    Returns the lookup table and the value ranges for incidence_angle, entropy, and alpha that are covered by the table.
    Invalid combinations of incidence_angle, entropy, and alpha (not covered by the X-Bragg model) result in eps = NaN.
    """
    theta_bounds = (np.radians(10.0), np.radians(60.0))
    eps_bounds = (2.0, 40.0)
    delta_bounds = (np.radians(0.0), np.radians(90.0))
    theta_grid, eps_grid, delta_grid = _get_xbragg_parameter_grids(theta_bounds, eps_bounds, delta_bounds)
    t = xbragg_model(theta_grid, eps_grid, delta_grid)
    entropy, anisotropy, alpha_mean, alpha_dominant = sarssm.h_a_alpha_decomposition(t)
    # construct lookup tables for each incidence angle
    entropy_bounds = (0.0, 0.8)
    alpha_bounds = (0.0, np.radians(35.0))
    incidence_slice_count = entropy.shape[0]
    lut_list = []
    for incidence_slice_index in range(incidence_slice_count):
        entropy_coords = entropy[incidence_slice_index].flatten()
        alpha_mean_coords = alpha_mean[incidence_slice_index].flatten()
        eps_vals = eps_grid[incidence_slice_index].flatten()
        eps_lut = _entropy_alpha_param_lut(
            entropy_coords,
            alpha_mean_coords,
            eps_vals,
            entropy_range=entropy_bounds,
            alpha_range=alpha_bounds,
            steps=256,
        )
        lut_list.append(eps_lut)
    eps_luts = np.array(lut_list)
    return eps_luts, theta_bounds, entropy_bounds, alpha_bounds


def coherency_matrix_to_xbragg_eps(t3, theta):
    """
    Estimate surface relative dielectrics from polarimetric coherency matrices using the inverted X-Bragg model.
    For each pixel, entropy and alpha values are obtained from the coherency matrix.
    Then, entropy, alpha, and the incidence angle (theta) are used to look up the corresponding eps.
    Parameters:
        t3 - coherency matrices, numpy array of shape (*B, 3, 3)
        theta - corresponding incidence angle in radians, numpy array of shape (*B,)
    Returns:
        Estimated surface dielectric constant (eps), numpy array of shape (*B,)
    """
    entropy, anisotropy, alpha, alpha_dominant = sarssm.h_a_alpha_decomposition(t3)
    eps_luts, theta_bounds, entropy_bounds, alpha_bounds = _create_xbragg_eps_lut()
    theta_num_slices, entropy_num_slices, alpha_num_slices = eps_luts.shape
    theta_indices, theta_invalid = _values_to_lut_indices(theta, theta_bounds, theta_num_slices)
    entropy_indices, entropy_invalid = _values_to_lut_indices(entropy, entropy_bounds, entropy_num_slices)
    alpha_indices, alpha_invalid = _values_to_lut_indices(alpha, alpha_bounds, alpha_num_slices)
    invalid = np.logical_or(np.logical_or(theta_invalid, entropy_invalid), alpha_invalid)
    inverted_eps = eps_luts[theta_indices, entropy_indices, alpha_indices]
    inverted_eps[invalid] = np.nan
    return inverted_eps
