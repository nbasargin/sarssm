"""
Interferometric soil moisture model for the coherence and phase triplets.

F. De Zan, A. Parizzi, P. Prats-Iraola and P. López-Dekker
A SAR Interferometric Model for Soil Moisture
IEEE Transactions on Geoscience and Remote Sensing, 2014
DOI: 10.1109/TGRS.2013.2241069
"""

import torch
from torch import nn
import sarssm


class Dezan2014InSARMoistureModel(nn.Module):
    def __init__(self, initial_sm_dict: dict, incidence, sand, clay, frequency):
        """
        Forward interferometric model for soil moisture implemented in PyTorch.
        The model predicts the interferometric parameters (complex coherence and phase triplets) given the soil moisture.
        The model works on a time series with N acquisitions, where each aquaition has its own id.
        Each acquisition must have the same number of points (same batch shape B).
        Model inversion can be performed by optimizing the soil moisture values to explain the observed interferometric parameters.
        Arguments:
            initial_sm - initial soil moisture, dict with N tensors of shape (*B,), dict keys are acquistion ids
            incidence - incidence angle, torch tensor of shape (*B,) or a scalar
            sand - soil sand content in percent (value from 0 to 100), torch tensor of shape (*B,) or a scalar
            clay - soil clay content in percent (value from 0 to 100), torch tensor of shape (*B,) or a scalar
            frequency - radar frequency in Hz, int or float
        """
        super().__init__()
        assert torch.is_tensor(incidence) and torch.is_tensor(sand) and torch.is_tensor(clay)
        for initial_sm_item in initial_sm_dict.values():
            assert torch.is_tensor(initial_sm_item)
        self.aquisition_ids = list(initial_sm_dict.keys())
        self.predicted_sm_dict = nn.ParameterDict(
            {key: nn.Parameter(initial_sm_item) for key, initial_sm_item in initial_sm_dict.items()}
        )
        self.frequency = frequency
        self.incidence = incidence
        self.sand = sand
        self.clay = clay

    def sm_to_soil_kz(self, sm):
        """
        Convert soil moisture to the vertical wavenumber in the soil (kz_soil) according to the model.
        Internal inputs:
            eps_r_soil - relative soil dielectrics, as predicted by Hallikainen model from the soil moisture
                         complex-valued, the imaginary part must be negative
            wavelength - radar wavelength in m
            incidence - incidence angle in radians
        Equations:
            kx^2 + kz_air^2 = omega^2 * mu_0 * eps_0  # Wave equation in air
            kx^2 + kz_soil^2 = omega^2 * mu_0 * eps_total_soil  # Wave equation in soil
            eps_total_soil = eps_r_soil * eps_0  # total soil dielectric constant = relative soil dielectrics * vacuum dielectrics
            f = c / wavelength  # center frequency = light speed / wavelength
            eps_0 = 1 / (mu_0 * c^2)  # vacuum dielectric permittivity
            mu_0 = 4 * pi * 10^-7  # vacuum magnetic permeability
            omega = 2 * pi * f  # angular frequency
            kx = 2 * pi / wavelength * sin(incidence)
            kz = 2 * pi / wavelength * cos(incidence)
        Obtaining kz_soil:
            kx^2 + kz_soil^2 = omega^2 * mu_0 * eps_total_soil  # Wave equation in soil
            kz_soil = sqrt( omega^2 * eps_total_soil * mu_0 - kx^2 )
            kz_soil = sqrt( omega^2 * eps_r_soil * eps_0 * mu_0 - kx^2 )
            kz_soil = sqrt( omega^2 * eps_r_soil * 1 / (mu_0 * c^2) * mu_0 - kx^2 )
            kz_soil = sqrt( omega^2 * eps_r_soil / (c^2) - kx^2 )
            kz_soil = sqrt( (2 * pi * f)^2 * eps_r_soil / (c^2) - kx^2 )
            kz_soil = sqrt( (2 * pi * c / wavelength)^2 * eps_r_soil / (c^2) - kx^2 )
            kz_soil = sqrt( (2 * pi / wavelength)^2 * eps_r_soil - kx^2 )
            kz_soil = sqrt( (2 * pi / wavelength)^2 * eps_r_soil - (2 * pi / wavelength * sin(incidence))^2 )
            kz_soil = sqrt( (2 * pi / wavelength)^2 * (eps_r_soil - sin^2(incidence)) )
            kz_soil = (2 * pi / wavelength) * sqrt( eps_r_soil - sin^2(incidence) )
        """
        light_speed = 299792458  # m/s
        wavelength = light_speed / self.frequency
        eps_r_soil = sarssm.moisture_to_eps_hallikainen(sm, sand=self.sand, clay=self.clay, frequency=self.frequency)
        kz_soil = torch.pi * 2 / wavelength * torch.sqrt(eps_r_soil - torch.sin(self.incidence) ** 2)
        # there are two solutions for sqrt, kz_soil with a negative imaginary part is physically correct
        if not torch.all(torch.isnan(kz_soil) | (kz_soil.imag < 0)):
            # imaginary part of kz_soil has the same sign as the imaginary part of eps_r_soil (which should be negative)
            raise ValueError("The imaginary part of kz_soil and eps_r_soil should be negative!")
        return kz_soil

    def kz_to_inf_coherence(self, kz_soil_1, kz_soil_2):
        """
        Conpute the expected interferometric coherence for two kz_soil values at different times.
        """
        return 2j * torch.sqrt(kz_soil_2.imag * kz_soil_1.imag) / (torch.conj(kz_soil_2) - kz_soil_1)

    def forward(self):
        """
        Forward model evaluation: given the moisture, compute all coherence pairs and phase triplets.
        Returns two dictionaries as a tuple: (coherence_dict, triplet_dict)
        Dictionary keys are tuples with aquisition ids (same ids as in the initial soil moisture dict)
            coherence_dict: contains the coherence for each pair of input soil moisture values
                            (aquisition_id_1, aquisition_id_2) -> complex coherence, torch tensor of shape (*B,)
                            for each pair, only one combination is included in the dict, e.g. (a, b) but not (b, a)
                            the reverse combination (b, a) can be obtanied by conjugating (a, b)
            triplet_dict: contains the phase triplet (real number between -pi and pi) for each triplet of moisture values
                          (aquisition_id_1, aquisition_id_2, aquisition_id_3) -> phase, torch tensor of shape (*B,)
        """
        num_aqusitions = len(self.aquisition_ids)
        kz_soil_dict = {key: self.sm_to_soil_kz(sm) for key, sm in self.predicted_sm_dict.items()}
        # compute all coherences
        coherence_dict = {}  # dict: tuple of aquisition_ids (id_i, id_j) -> coherence tensor
        for i in range(num_aqusitions):
            for j in range(i + 1, num_aqusitions):
                id_i, id_j = self.aquisition_ids[i], self.aquisition_ids[j]
                kz_soil_i, kz_soil_j = kz_soil_dict[id_i], kz_soil_dict[id_j]
                coherence_dict[id_i, id_j] = self.kz_to_inf_coherence(kz_soil_i, kz_soil_j)
        # compute all triplets
        triplet_dict = {}
        for i in range(num_aqusitions):
            for j in range(i + 1, num_aqusitions):
                for k in range(j + 1, num_aqusitions):
                    id_i, id_j, id_k = self.aquisition_ids[i], self.aquisition_ids[j], self.aquisition_ids[k]
                    coherence_ij = coherence_dict[(id_i, id_j)]
                    coherence_jk = coherence_dict[(id_j, id_k)]
                    coherence_ki = torch.conj(coherence_dict[(id_i, id_k)])  # (k, i) is like conj(i, k)
                    triplet_ijk = coherence_ij * coherence_jk * coherence_ki
                    triplet_dict[id_i, id_j, id_k] = torch.angle(triplet_ijk)
        return coherence_dict, triplet_dict


def moisture_to_insar_coherence(moisture_1, moisture_2, incidence, sand, clay, frequency):
    """
    Predict the expected complex interferometric coherence given two soil moisture values at different times.
    Numpy wrapper around the Dezan2014InSARMoistureModel. Use the model directly to compute coherence for
    multiple moisture combinations at once.
    Arguments:
        moisture_1: first soil moisture value, numpy array of shape (*B,), where B are the batch dimensions
        moisture_2: second moisture value, numpy array of shape (*B,)
        incidence: incidence angle in radians, numpy array of shape (*B,) or a scalar
        sand: soil sand content in percent (value from 0 to 100), numpy array of shape (*B,) or a scalar
        clay: soil clay content in percent (value from 0 to 100), numpy array of shape (*B,) or a scalar
        frequency: radar frequency in Hz, int or float
    Returns:
        complex_coherence between two acquisitions predicted by the model, complex-valued numpy array of shape (*B,)
    """
    moisture_dict = {
        "moisture_1": torch.tensor(moisture_1),
        "moisture_2": torch.tensor(moisture_2),
    }
    incidence = torch.tensor(incidence)
    sand = torch.tensor(sand)
    clay = torch.tensor(clay)
    model = Dezan2014InSARMoistureModel(
        initial_sm_dict=moisture_dict, incidence=incidence, sand=sand, clay=clay, frequency=frequency
    )
    coherence_dict, triplet_dict = model.forward()
    complex_coherence = coherence_dict[("moisture_1", "moisture_2")]
    return complex_coherence.cpu().detach().numpy()


def moisture_to_insar_phase_triplet(moisture_1, moisture_2, moisture_3, incidence, sand, clay, frequency):
    """
    Predict the expected interferometric phase triplet given three soil moisture values at different times.
    Numpy wrapper around the Dezan2014InSARMoistureModel. Use the model directly to compute multiple coherences at once.
    Arguments:
        moisture_1: first soil moisture value, numpy array of shape (*B,), where B are the batch dimensions
        moisture_2: second moisture value, numpy array of shape (*B,)
        moisture_3: third moisture value, numpy array of shape (*B,)
        incidence: incidence angle in radians, numpy array of shape (*B,) or a scalar
        sand: soil sand content in percent (value from 0 to 100), numpy array of shape (*B,) or a scalar
        clay: soil clay content in percent (value from 0 to 100), numpy array of shape (*B,) or a scalar
        frequency: radar frequency in Hz, int or float
    Returns:
        phase_triplet: phase in radians predicted by the model, real-valued numpy array of shape (*B,)
    """
    moisture_dict = {
        "moisture_1": torch.tensor(moisture_1),
        "moisture_2": torch.tensor(moisture_2),
        "moisture_3": torch.tensor(moisture_3),
    }
    incidence = torch.tensor(incidence)
    sand = torch.tensor(sand)
    clay = torch.tensor(clay)
    model = Dezan2014InSARMoistureModel(
        initial_sm_dict=moisture_dict, incidence=incidence, sand=sand, clay=clay, frequency=frequency
    )
    coherence_dict, triplet_dict = model.forward()
    phase_triplet = triplet_dict[("moisture_1", "moisture_2", "moisture_3")]
    return phase_triplet.cpu().detach().numpy()
