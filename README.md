# Automobile Insurance Fraud Detector using Generative Adversarial Networks

This project analyzes fraudulent vehicle insurance claims using the Kaggle [Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection?select=fraud_oracle.csv) dataset and explores whether Generative Adversarial Networks (GANs) can help model and detect fraud patterns within imbalanced datasets containing little fraudulent cases.

The project follows three main stages:

### 1. Data Cleaning & Exploration
- Removed irrelevant temporal identifiers and corrected inconsistent policy information.
- Encoded categorical variables and standardized numerical variables.
- Split data into fraudulent vs. non-fraudulent subsets, which were then saved as csv for future use.
- Produced visualisations, e.g. correlation heatmaps and difference heatmaps, to compare structural patterns between the fraudulent and non-fraudulent groups.

### 2. Generative Modeling (leveraging CTGAN)
- Use SDV’s **CTGANSynthesizer** to learn the distribution of claim data.
- Train CTGAN to generate realistic synthetic fraudulent claims.
- Evaluate the fidelity of generated data through statistical comparisons and visual checks.

### 3. Fraud Detection
- Use the trained CTGAN to create synthetic data.
- Assess whether synthetic data created by CTGAN improves fraud detection.
