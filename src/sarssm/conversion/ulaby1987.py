"""
Conversion from the gravimetric moisture of corn leaves to the complex relative dielectric constant.

Fawwaz T. Ulaby and Mohamed A. El-Rayes
Microwave Dielectric Spectrum of Vegetation Part II: Dual-Dispersion Model
IEEE Transactions on Geoscience and Remote Sensing, September 1987
DOI: 10.1109/TGRS.1987.289833
"""


def corn_moisture_to_eps_ulaby(m_g, frequency):
    """
    m_g: gravimetric moisture content of corn leaves, the model is valid for values in [0.05, 0.7]
    frequency_ghz: frequency in Hz
    """
    frequency_ghz = frequency * 1e-9
    # equation (17): ionic conductivity of the aqueous solution (in siemens per meter)
    sigma = 1.27
    # equation (2): dielectric constant of free water, salinity below 0.01, room temperature (22°C)
    eps_f = 4.9 + 75 / (1 + 1j * frequency_ghz / 18) - 1j * 18 * sigma / frequency_ghz
    # equation (8): dielectric constant of the bulk vegetation-bound water mixture
    eps_b = 2.9 + 55 / (1 + (1j * frequency_ghz / 0.18) ** 0.5)
    # equation (14): nondispersive residual dielectrics
    eps_r = 1.7 - 0.74 * m_g + 6.16 * m_g**2
    # equation (15): volume fraction of free water
    v_fw = m_g * (0.55 * m_g - 0.076)
    # equation (16): volume fraction of the bulk vegetation-bound water mixture
    v_b = 4.64 * m_g**2 / (1 + 7.36 * m_g**2)
    # equation (9) / (10): plant dielectrics as sum of three components (residual + free water + bound water)
    return eps_r + v_fw * eps_f + v_b * eps_b
