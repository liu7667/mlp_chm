# mlp_chm
Canopy Height Inversion of Eucalyptus Plantations in Guangxi Based on Multi-Source Remote Sensing Data
# MLP-Based Canopy Height Inversion of Eucalyptus Plantations in Guangxi, China

This repository provides the core implementation of a multi-source remote sensing based Multi-Layer Perceptron (MLP) model for canopy height inversion of eucalyptus plantations in Guangxi, China.

The code is associated with the manuscript:

> **Canopy Height Inversion of Eucalyptus Plantations in Guangxi Based on Multi-Source Remote Sensing Data**

The study integrates GEDI LiDAR observations with Sentinel-1, Sentinel-2, and DEM data to estimate canopy height of eucalyptus plantations under complex terrain and high vegetation coverage conditions.

---

## 1. Overview

Accurate estimation of forest canopy height is important for forest resource monitoring, plantation management, biomass estimation, and carbon stock assessment.

Eucalyptus plantations in Guangxi are widely distributed across areas with considerable differences in terrain, forest stand structure, and vegetation coverage. These conditions may introduce uncertainties into canopy height estimation based solely on optical remote sensing data or conventional spatial-context models.

In this study, GEDI LiDAR observations are used as reference canopy height measurements, while multi-source remote sensing variables from Sentinel-1, Sentinel-2, and SRTM DEM are used as predictor variables.

A residual-enhanced MLP model is developed to establish the nonlinear relationship between GEDI canopy height and the corresponding multi-source remote sensing features.

The main characteristics of the proposed method are:

- Integration of GEDI, Sentinel-1, Sentinel-2, and DEM data;
- Pixel-wise regression based on GEDI footprint–pixel correspondence;
- Residual-enhanced MLP architecture for nonlinear regression;
- Application to eucalyptus plantations in Guangxi, China;
- Evaluation against several representative machine learning and deep learning models.

---

## 2. Study Area

The study focuses on eucalyptus plantations in three representative regions of Guangxi, China:

- Chongzuo
- Beihai
- Hezhou

These areas represent different terrain and plantation conditions, including mountainous and hilly terrain as well as relatively flat coastal areas.

The spatial heterogeneity of terrain and vegetation conditions provides a suitable test environment for evaluating canopy height inversion using multi-source remote sensing data.

---

## 3. Data

### 3.1 GEDI LiDAR

GEDI Level 2A data are used to obtain canopy relative height metrics.

The main target variable used in this study is:

- **RH95**

GEDI observations are subjected to quality control before being used for model development.

The main quality-control criteria include:

- `quality_flag = 1`
- `degrade_flag = 0`
- Sensitivity ≥ 0.95
- Terrain slope < 25°

GEDI observations are spatially matched with the corresponding remote sensing features for model training and validation.

---

### 3.2 Sentinel-2

Sentinel-2 surface reflectance data are used to provide optical spectral information.

The main spectral variables used by the final model include:

- B3
- B4
- B5
- B6
- B8
- B11
- B12

Vegetation indices are also calculated from Sentinel-2 data:

- NDVI
- EVI
- NDRE
- GNDVI
- SAVI

Sentinel-2 observations are processed using cloud and cloud-shadow masking followed by monthly compositing.

---

### 3.3 Sentinel-1

Sentinel-1 GRD data are used to provide microwave backscatter information.

The final model uses:

- VV
- VH

Sentinel-1 observations are processed and composited on a monthly basis before being matched with the optical and terrain features.

---

### 3.4 DEM

SRTM DEM data are used to represent terrain conditions.

The final model uses:

- Elevation

Terrain information is included to account for the influence of topographic variation on canopy height estimation.

---

## 4. Input Features

The final model directly uses the multi-source remote sensing features constructed in the study.

No additional feature-selection procedure is applied separately for the different models.

The final input consists of **15 features**:

| Category | Feature |
|---|---|
| Sentinel-2 spectral | B3 |
| Sentinel-2 spectral | B4 |
| Sentinel-2 spectral | B5 |
| Sentinel-2 spectral | B6 |
| Sentinel-2 spectral | B8 |
| Sentinel-2 spectral | B11 |
| Sentinel-2 spectral | B12 |
| Vegetation index | NDVI |
| Vegetation index | EVI |
| Vegetation index | NDRE |
| Vegetation index | GNDVI |
| Vegetation index | SAVI |
| Sentinel-1 SAR | VV |
| Sentinel-1 SAR | VH |
| Terrain | Elevation |

All input features are standardized before model training.

---

## 5. Data Organization

For each valid GEDI sample, the corresponding multi-source remote sensing data are spatially matched.

A 64 × 64 pixel window is used for unified sample organization.

At a spatial resolution of 10 m, the window corresponds to approximately:

**640 m × 640 m**

The 64 × 64 window is mainly used to preserve the GEDI sample and its surrounding environmental information and to provide a consistent spatial data structure for model comparison.

For the pixel-wise MLP model, the actual regression input is the multi-source feature vector corresponding to the GEDI footprint center rather than the entire 64 × 64 neighborhood.

---

## 6. Proposed Model

The proposed method is a residual-enhanced Multi-Layer Perceptron (MLP).

The model establishes a nonlinear mapping:

GEDI canopy height

→

multi-source remote sensing features

→

predicted canopy height

The residual structure is introduced to improve feature representation and information propagation within the MLP.

The final model configuration is:

| Parameter | Value |
|---|---:|
| Input features | 15 |
| Hidden architecture | 256–256 |
| Dropout rate | 0.2 |
| Learning rate | 0.0001 |
| Batch size | 16 |
| Loss function | Huber loss |
| Optimizer | AdamW |

The parameters were determined through controlled parameter experiments involving dropout rate, learning rate, network architecture, and batch size.

---

## 7. Model Comparison

To evaluate the effectiveness of the proposed model, several representative machine learning and deep learning models are compared under the same experimental framework.

The compared models include:

- MLP
- ResNet
- ResNet-18
- CNN
- U-Net
- DenseNet
- Random Forest (RF)

The models use the same GEDI samples, training/validation partition, preprocessing procedure, and source feature variables.

For convolutional neural networks, 64 × 64 multi-channel image patches are used as spatial inputs.

For MLP and Random Forest, the multi-source features corresponding to the GEDI footprint center are used as the input feature vector.

---

## 8. Model Performance

The main validation results are summarized below.

| Model | R² | RMSE (m) | MAE (m) | rRMSE |
|---|---:|---:|---:|---:|
| MLP | 0.56 | 6.68 | 4.46 | 37% |
| ResNet | 0.55 | 7.40 | 5.35 | 41% |
| ResNet-18 | 0.54 | 6.76 | 4.96 | 37% |
| CNN | 0.54 | 7.65 | 5.54 | 42% |
| U-Net | 0.51 | 7.91 | 5.73 | 44% |
| DenseNet | 0.46 | 7.54 | 5.65 | 42% |
| Random Forest | 0.42 | 8.26 | 6.17 | 45% |

The MLP model achieved the highest coefficient of determination (R²) and the lowest RMSE and MAE among the compared models under the experimental conditions of this study.

The results suggest that the pixel-wise regression strategy can effectively establish the nonlinear relationship between GEDI canopy height and multi-source remote sensing features for the studied eucalyptus plantations.

---

## 9. Repository Structure

The recommended repository structure is:

```text
mlp_chm/
│
├── README.md
│
├── code/
│   ├── model.py
│   ├── train_mlp.py
│   ├── predict_chm.py
│   └── evaluate.py
│
├── requirements.txt
│
└── docs/
    └── methodology.md
