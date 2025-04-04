"""
Model inversion for the interferometric soil moisture model.

F. De Zan, A. Parizzi, P. Prats-Iraola and P. López-Dekker
A SAR Interferometric Model for Soil Moisture
IEEE Transactions on Geoscience and Remote Sensing, 2014
DOI: 10.1109/TGRS.2013.2241069
"""

import numpy as np
import torch
import sarssm


def slc_to_insar_parameters(slc_dict, window_size):
    """
    SLCs to interferometric parameters (complex coherence and phase triplets).
    Input:
        slc_dict: dict with N SLC images of shape (Az, Rg), dict keys are acquistion ids
        window_size: tuple (az_window, rg_window) to compute coherences and triplets
    Returns two dictionaries as a tuple: (coherence_dict, triplet_dict)
        coherence_dict: contains the complex coherence for each pair of input SLCs
                        (aquisition_id_1, aquisition_id_2) -> complex coherence, numpy array of shape (Az, Rg)
                        for each pair, only one combination is included in the dict, e.g. (a, b) but not (b, a)
                        the reverse combination (b, a) can be obtanied by conjugating (a, b)
        triplet_dict: contains the phase triplet (real number between -pi and pi) for each triplet of SLCs
                        (aquisition_id_1, aquisition_id_2, aquisition_id_3) -> phase, numpy of shape (Az, Rg)
    """
    aquisition_ids = list(slc_dict.keys())
    num_aqusitions = len(aquisition_ids)
    # coherences
    coherence_dict = {}  # dict: tuple of aquisition_ids (id_i, id_j) -> coherence array
    for i in range(num_aqusitions):
        for j in range(i + 1, num_aqusitions):
            id_i, id_j = aquisition_ids[i], aquisition_ids[j]
            slc_i, slc_j = slc_dict[id_i], slc_dict[id_j]
            coherence_ij = sarssm.complex_coherence(slc_i, slc_j, window_size)
            coherence_dict[id_i, id_j] = coherence_ij
    # triplets
    triplet_dict = {}  # Map index tuple (i, j, k) to the triplet array
    for i in range(num_aqusitions):
        for j in range(i + 1, num_aqusitions):
            for k in range(j + 1, num_aqusitions):
                id_i, id_j, id_k = aquisition_ids[i], aquisition_ids[j], aquisition_ids[k]
                coherence_ij = coherence_dict[id_i, id_j]
                coherence_jk = coherence_dict[id_j, id_k]
                coherence_ki = np.conj(coherence_dict[id_i, id_k])  # (k, i) is like conj(i, k)
                triplet_ijk = coherence_ij * coherence_jk * coherence_ki
                triplet_dict[id_i, id_j, id_k] = np.angle(triplet_ijk)
    return coherence_dict, triplet_dict


def _np_dict_to_torch(np_dict):
    """
    Dictionary with numpy arrays -> dictionary with torch tensors.
    """
    return {key: torch.tensor(np_array) for key, np_array in np_dict.items()}


def _abs_on_dict(torch_dict):
    return {key: torch.abs(complex_data) for key, complex_data in torch_dict.items()}


def insar_parameters_to_moisture(
    coherence_dict, triplet_dict, initial_sm_dict, known_sm_aquisition_id, incidence, sand, clay, frequency, iters=320
):
    """
    Invert the De Zan soil moisture model and predict soil moisture from interferometric parameters (coherence and triplets).
    The provided initial soil moisture values are iteratively adjusted until a combination is found that best explains the
    observed interferometric parameters. In addition, one soil moisture value must be known.
    The model works on a time series with N acquisitions, where each aquaition has its own id.
    Each acquisition must have the same number of points (same batch shape B).
    Input:
        coherence_dict - dictionary that contains the observed complex coherence values
                         (aquisition_id_1, aquisition_id_2) -> complex coherence, numpy array of shape (*B,)
        triplet_dict - dictionary that contains the observed triplet values (real numbers between -pi and pi)
                       (aquisition_id_1, aquisition_id_2, aquisition_id_3) -> phase, numpy of shape (*B,)
        initial_sm_dict - dictionary that contains the initial soil moisture values (between 0 and 1) for each acquisition
                                aquisition_id -> initial soil moisture, numpy array of shape (*B,)
        known_sm_aquisition_id - the acquisition id of the moisture values that are assumed to be known
                                 those values are kept constant during the optimization
        incidence - incidence angle, numpy array of shape (*B,) or a scalar
        sand - soil sand content in percent (value from 0 to 100), numpy array of shape (*B,) or a scalar
        clay - soil clay content in percent (value from 0 to 100), numpy array of shape (*B,) or a scalar
        frequency - radar frequency in Hz, int or float
        iters - number of model fitting iterations
    Returns:
        the fitted model
    """
    # form torch tensors
    obs_abs_coh_dict = _abs_on_dict(_np_dict_to_torch(coherence_dict))
    obs_tri_dict = _np_dict_to_torch(triplet_dict)
    initial_sm_dict = _np_dict_to_torch(initial_sm_dict)
    incidence = torch.tensor(incidence, dtype=torch.float32)
    sand = torch.tensor(sand, dtype=torch.float32)
    clay = torch.tensor(clay, dtype=torch.float32)
    # construct model and fix the known moisture value
    model = sarssm.Dezan2014InSARMoistureModel(
        initial_sm_dict=initial_sm_dict, incidence=incidence, sand=sand, clay=clay, frequency=frequency
    )
    model.predicted_sm_dict[known_sm_aquisition_id].requires_grad = False  # known moisture, will not be updated

    # the loss combines the deviation of all coherences and triplets
    def _criterion(model_coh_dict: dict, model_tri_dict: dict, obs_coh_dict: dict, obs_tri_dict: dict):
        triplet_weight = 1
        coh_losses = torch.stack(
            [torch.sum((model_coh_dict[key] - obs_coh_dict[key]) ** 2) for key in model_coh_dict.keys()]
        )
        # ignore phase wraps, triplet phase is often close to 0, wraps unlikely
        tri_losses = torch.stack(
            [torch.sum((model_tri_dict[key] - obs_tri_dict[key]) ** 2) for key in model_tri_dict.keys()]
        )
        return torch.sum(coh_losses) + triplet_weight * torch.sum(tri_losses)

    betas = (0.975, 0.995)
    lr = 0.05
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": lr}], betas=betas)
    for iteration in range(iters):
        optimizer.zero_grad()
        model_coh_dict, model_tri_dict = model.forward()
        model_abs_coh_dict = _abs_on_dict(model_coh_dict)
        loss = _criterion(model_abs_coh_dict, model_tri_dict, obs_abs_coh_dict, obs_tri_dict)
        loss.backward()
        optimizer.step()
        # clamp soil moisture
        with torch.no_grad():
            min_sm = 0.03
            max_sm = 0.70
            for predicted_sm in model.predicted_sm_dict.values():
                predicted_sm[predicted_sm < min_sm] = min_sm
                predicted_sm[predicted_sm > max_sm] = max_sm
    return model
