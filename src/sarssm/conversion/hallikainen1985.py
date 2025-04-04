"""
Conversion from the soil moisture to the complex-valued relative dielectric constant.
The soil texture (sand, silt, clay content) and the microwave frequency are taken into account.

M. T. Hallikainen, F. T. Ulaby, M. C. Dobson, M. A. El-rayes and L.-k. Wu
Microwave Dielectric Behavior of Wet Soil-Part 1: Empirical Models and Experimental Observations
IEEE Transactions on Geoscience and Remote Sensing, 1985
DOI: 10.1109/TGRS.1985.289497
"""


def moisture_to_eps_hallikainen(mv, sand, clay, frequency):
    """
    Conversion from the soil moisture to the complex-valued relative dielectric constant.
    The soil texture (sand, silt, clay content) and the microwave frequency are taken into account.
    The empirical Hallikainen model provides conversion for specific microwave frequencies:
        1.4 GHz (L-band), 4 GHz (S-band), 6 GHz (C-band), 8 GHz, 10 GHz (X-band), 12 GHz, 14 GHz, 16 GHz, and 18 GHz.
        For the conversion, the closest available frequency is used.
    Parameters:
        mv - volumetric soil moisture (value range 0 to 1)
        sand - sand textural component, in percent (value range 0 to 100)
        clay - clay textural component, in percent (value range 0 to 100)
        frequency - frequency in Hz, (allowed value range 1e9 to 20e9)
    Returns:
        complex-valued relative dielectric constant, the imaginary part is negative
    """
    # find the closest available frequency
    if 1_000_000_000 <= frequency < 2_700_000_000:
        frequency = 1_400_000_000  # L-band - F-SAR: 1.325 GHz; ALOS PALSAR: 1.27 GHz
    elif 2_700_000_000 <= frequency < 5_000_000_000:
        frequency = 4_000_000_000  # S-band - F-SAR: 3.25 GHz;
    elif 5_000_000_000 <= frequency < 7_000_000_000:
        frequency = 6_000_000_000  # C-band - F-SAR: 5.3 GHz; Sentinel-1: 5.405 GHz
    elif 7_000_000_000 <= frequency < 9_000_000_000:
        frequency = 8_000_000_000
    elif 9_000_000_000 <= frequency < 11_000_000_000:
        frequency = 10_000_000_000  # X-band - F-SAR: 9.6 GHz; TerraSAR-X: 9.65 GHz
    elif 11_000_000_000 <= frequency < 13_000_000_000:
        frequency = 12_000_000_000
    elif 13_000_000_000 <= frequency < 15_000_000_000:
        frequency = 14_000_000_000
    elif 15_000_000_000 <= frequency < 17_000_000_000:
        frequency = 16_000_000_000
    elif 17_000_000_000 <= frequency <= 20_000_000_000:
        frequency = 18_000_000_000
    else:
        raise ValueError("Frequency must be between 1 and 20 GHz")
    # real coefficients: (a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1, c_2)
    real_coeff = {
        1_400_000_000: (2.862, -0.012, 0.001, 3.803, 0.462, -0.341, 119.006, -0.500, 0.633),
        4_000_000_000: (2.927, -0.012, -0.001, 5.505, 0.371, 0.062, 114.826, -0.389, -0.547),
        6_000_000_000: (1.993, 0.002, 0.015, 38.086, -0.176, -0.633, 10.720, 1.256, 1.522),
        8_000_000_000: (1.997, 0.002, 0.018, 25.579, -0.017, -0.412, 39.793, 0.723, 0.941),
        10_000_000_000: (2.502, -0.003, -0.003, 10.101, 0.221, -0.004, 77.482, -0.061, -0.135),
        12_000_000_000: (2.200, -0.001, 0.012, 26.473, 0.013, -0.523, 34.333, 0.284, 1.062),
        14_000_000_000: (2.301, 0.001, 0.009, 17.918, 0.084, -0.282, 50.149, 0.012, 0.387),
        16_000_000_000: (2.237, 0.002, 0.009, 15.505, 0.076, -0.217, 48.260, 0.168, 0.289),
        18_000_000_000: (1.912, 0.007, 0.021, 29.123, -0.190, -0.545, 6.960, 0.822, 1.195),
    }
    # imag coefficients: (x_0, x_1, x_2, y_0, y_1, y_2, z_0, z_1, z_2)
    imag_coeff = {
        1_400_000_000: (0.356, -0.003, -0.008, 5.507, 0.044, -0.002, 17.753, -0.313, 0.206),
        4_000_000_000: (0.004, 0.001, 0.002, 0.951, 0.005, -0.010, 16.759, 0.192, 0.290),
        6_000_000_000: (-0.123, 0.002, 0.003, 7.502, -0.058, -0.116, 2.942, 0.452, 0.543),
        8_000_000_000: (-0.201, 0.003, 0.003, 11.266, -0.085, -0.155, 0.194, 0.584, 0.581),
        10_000_000_000: (-0.070, 0.000, 0.001, 6.620, 0.015, -0.081, 21.578, 0.293, 0.332),
        12_000_000_000: (-0.142, 0.001, 0.003, 11.868, -0.059, -0.225, 7.817, 0.570, 0.801),
        14_000_000_000: (-0.096, 0.001, 0.002, 8.583, -0.005, -0.153, 28.707, 0.297, 0.357),
        16_000_000_000: (-0.027, -0.001, 0.003, 6.179, 0.074, -0.086, 34.126, 0.143, 0.206),
        18_000_000_000: (-0.071, 0.000, 0.003, 6.938, 0.029, -0.128, 29.945, 0.275, 0.377),
    }
    # convert moisture to complex dielectrics
    a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1, c_2 = real_coeff[frequency]
    eps_real = (
        (a_0 + a_1 * sand + a_2 * clay)
        + (b_0 + b_1 * sand + b_2 * clay) * mv
        + (c_0 + c_1 * sand + c_2 * clay) * (mv**2)
    )
    x_0, x_1, x_2, y_0, y_1, y_2, z_0, z_1, z_2 = imag_coeff[frequency]
    eps_imag = (
        (x_0 + x_1 * sand + x_2 * clay)
        + (y_0 + y_1 * sand + y_2 * clay) * mv
        + (z_0 + z_1 * sand + z_2 * clay) * (mv**2)
    )
    return eps_real - 1j * eps_imag
