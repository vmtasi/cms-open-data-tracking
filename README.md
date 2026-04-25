# CMS Open Data Tracking: Physics-Informed ML Benchmarking

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-f7931e.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a framework for benchmarking machine learning algorithms against traditional linear models for **charged particle trajectory recovery** in high-energy physics. 

Using kinematic seeds extracted from **CMS Open Data**, this project implements a hybrid stochastic simulation to model muon propagation through a 13-layer detector. The core objective is to evaluate the extrapolation capabilities of Deep Learning (MLPs) versus Linear Regression in the presence of material scattering, energy loss, and detector inefficiencies.


## Overview

Modern high-energy physics relies heavily on sequential filtering algorithms (like the Kalman Filter) for particle tracking. As particle density increases exponentially with upgrades like the HL-LHC, Machine Learning is being explored as a highly parallelizable alternative. 

This repository establishes a rigorous **Negative Control Baseline**. It proves experimentally that for standard, high-momentum signal muons ($p_T > 10$ GeV), linear extrapolation is statistically sufficient, and Deep Learning offers no distinct advantage. This validates the linearity of the high-$p_T$ regime and sets the stage for applying ML to the highly non-linear low-$p_T$ (pileup) regime.

## Key Features

1. **Real-Seeded Hybrid Simulation:**
   - Extracts true 4-vectors ($p_T, \eta, \phi, q$) from CMS Open Data ROOT files.
   - Preserves the true phase space and momentum biases of the CMS detector.

2. **Stochastic Physics Engine:**
   - **Lorentz Propagation:** Helical motion in a $3.8\text{ T}$ solenoidal magnetic field.
   - **Process Noise:** Iterative injection of Multiple Coulomb Scattering ($\theta \propto 1/p_T$) and continuous energy loss ($dE/dx$).

3. **Imperfect Detector Modeling:**
   - Simulates 13 tracker layers (Pixel + Strip).
   - Injects realistic Gaussian measurement noise ($\sigma_{xy} \approx 150 \mu m$, $\sigma_{z} \approx 500 \mu m$).
   - Implements a **10% Dead Channel** probability to force model robustness against sparse data.

## The "Gap-Strided" Challenge

To rigorously test whether models are learning second-derivative physics (curvature) or simply memorizing nearest-neighbor interpolations, the dataset is structured as a **Blind Extrapolation Task**:

* **Input:** A sequence of 7 noisy hits (the History).
* **The Gap:** A 2-layer blind spot ($\sim 20\text{ cm}$ of unmeasured flight).
* **Target:** The true physical coordinates of the 2 hits *after* the gap.

This forces the models to project the trajectory across empty space, heavily penalizing naive linear models on curved tracks.

## Findings

The models were evaluated on the final 3D point prediction after crossing the blind gap.

| Model | Final Point RMSE (meters) |
| :--- | :--- |
| **Linear Regression** | `~ 0.30 m` |
| **Deep Neural Network (MLP)** | `~ 0.31 m` |
| **Random Forest** | `~ 0.50 m` |

**Conclusion:** The performance parity between Linear Regression and the MLP confirms that the standard CMS Open Data regime is effectively linear. The curvature of high-$p_T$ signal muons is mathematically negligible over short distances, validating the historical use of linear estimators in this specific domain.

## Installation & Usage

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Setup
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/vmtasi/cms-open-data-tracking.git](https://github.com/vmtasi/cms-open-data-tracking.git)
cd cms-open-data-tracking
pip install -r requirements.txt
