# SAR SSM package

Selected algorithms for surface soil moisture (SSM) estimation from SAR data.


## Installation 

`pip install sarssm`

Alternatively, you can install the package in editable mode:
- clone this repository
- activate a virtual environment (e.g. conda)
- install this package (in editable mode) with `pip install -e ./sarssm/`


## List of relevant papers
The code is based on the following papers.


### Dielectrics to Moisture Conversion
```
M. T. Hallikainen, F. T. Ulaby, M. C. Dobson, M. A. El-rayes, L.-k. Wu
Microwave Dielectric Behavior of Wet Soil-Part 1: Empirical Models and Experimental Observations
IEEE Transactions on Geoscience and Remote Sensing, 1985
DOI: 10.1109/TGRS.1985.289497
```

```
G. C. Topp, J. L. Davis, A. P. Annan
Electromagnetic determination of soil water content: Measurements in coaxial transmission lines
Water Resources Research, 1980
DOI: 10.1029/WR016i003p00574
```

```
F. T. Ulaby, M. A. El-Rayes
Microwave Dielectric Spectrum of Vegetation Part II: Dual-Dispersion Model
IEEE Transactions on Geoscience and Remote Sensing, 1987
DOI: 10.1109/TGRS.1987.289833
```

```
V. L. Mironov, L. G. Kosolapova, S. V. Fomin
Physically and Mineralogically Based Spectroscopic Dielectric Model for Moist Soils
IEEE Transactions on Geoscience and Remote Sensing, 2009
DOI: 10.1109/TGRS.2008.2011631
```


### PolSAR
```
I. Hajnsek, E. Pottier, S. R. Cloude
Inversion of surface parameters from polarimetric SAR
IEEE Transactions on Geoscience and Remote Sensing, 2003
DOI: 10.1109/TGRS.2003.810702
```

```
S. R. Cloude, E. Pottier
A review of target decomposition theorems in radar polarimetry
IEEE Transactions on Geoscience and Remote Sensing, 1996
DOI: 10.1109/36.485127
```


### InSAR
```
F. De Zan, A. Parizzi, P. Prats-Iraola, P. López-Dekker
A SAR Interferometric Model for Soil Moisture
IEEE Transactions on Geoscience and Remote Sensing, Jan. 2014
DOI: 10.1109/TGRS.2013.2241069
```


## Changelog

### v1.1.0
- `moisture_to_eps_mironov` added: new function to convert soil moisture to dielectrics
- `Dezan2014InSARMoistureModel` (and related utility wrappers) can now choose the conversion algorithm with the `sm2eps_alg` parameter
- `insar_parameters_to_moisture` can now choose the conversion algorithm with the `sm2eps_alg` parameter

### v1.0.0
- initial release
