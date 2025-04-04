"""
Conversion from the relative dielectric constant to soil moisture.

G. C. Topp, J. L. Davis, A. P. Annan
Electromagnetic determination of soil water content: Measurements in coaxial transmission lines
Water Resources Research, 1980
DOI: 10.1029/WR016i003p00574
"""


def eps_to_moisture_topp(eps_r):
    """
    Conversion from the relative dielectric constant to soil moisture.
    Parameters:
        eps_r: relative dielectric constant of the soil (typically the real part), valid values range from 2 to about 50
    Returns:
        volumetric soil moisture vith values ranging between 0 (0%) and 1 (100%).
    """
    return -5.3e-2 + 2.92e-2 * eps_r - 5.5e-4 * eps_r**2 + 4.3e-6 * eps_r**3
