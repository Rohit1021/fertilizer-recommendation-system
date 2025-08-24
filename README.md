# Fertilizer Recommendation System(Kaggle Playground Series)

A machine learning–powered web application that predicts the **Top-3 recommended fertilizers** for given soil, crop, and environmental conditions.  
Built using the **Fertilizer Recommendation Dataset** and extended with a **Flask web app** for interactive predictions.

---

## Problem Statement
Farmers need guidance on which fertilizer to use based on crop type, soil conditions, and environmental parameters.  
The objective is to build a model that learns from past labeled data and recommends the **most suitable fertilizers**, evaluated using **Mean Average Precision @ 3 (MAP@3)**.

---

## Dataset
Each row represents a field condition with the recommended fertilizer:

| id | Temperature | Humidity | Moisture | Soil Type | Crop Type | Nitrogen | Potassium | Phosphorous | Fertilizer Name |
|----|-------------|----------|----------|-----------|-----------|----------|-----------|-------------|-----------------|

- **Categorical features:** `Soil Type`, `Crop Type`  
- **Numeric features:** `Temperature`, `Humidity`, `Moisture`, `Nitrogen`, `Potassium`, `Phosphorous`  
- **Target:** `Fertilizer Name`  

---

## Features
- **Data Preprocessing**
  - Label encoding for categorical features.
  - Missing numerics imputed with **training medians**.
  - Consistent schema management (`schema.json`).
- **Modeling**
  - **XGBoost Classifier** trained with stratified K-Fold cross-validation.
  - Predictions return **Top-3 fertilizers** with associated probabilities.
- **Web Application**
  - Flask backend with dropdowns for categorical inputs and text boxes for numeric features.
  - Displays **Top-3 fertilizer recommendations** directly in the browser.
- **Deployment Ready**
  - Schema-driven inference pipeline ensures consistency between training and prediction.
  - Simple to deploy on Render / Railway / HuggingFace Spaces.

---

## Tech Stack
- **Python 3.8+**
- **Libraries:** XGBoost, scikit-learn, pandas, numpy, Flask
