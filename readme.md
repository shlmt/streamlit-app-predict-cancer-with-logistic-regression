# Cancer Diagnostic Web Application

An end-to-end Machine Learning web application that predicts whether a tumor is **Benign** or **Malignant** based on clinical features.

## Repository Overview
This repository contains the complete pipeline:
* **Pipeline & Training:** Data loading, preprocessing, scaling, and model training.
* **Inference & UI:** Production-ready Streamlit web application for real-time prediction and interactive data visualization.

## Dataset
The model is trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset** from UCI, available on [Kaggle](https://www.kaggle.com/datasets/abhinavmangalore/breast-cancer-dataset-wisconsin-diagnostic-uci?resource=download). It includes 30 features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image (such as radius, texture, perimeter, and area).

## Tech Stack
* **Core:** Python
* **Machine Learning:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Visualization:** Plotly (Radar charts & gauges)

---

## Core Algorithm: Logistic Regression

**Logistic Regression** is a fundamental algorithm for **Classification** tasks. It models the probability of a binary outcome (true or false, healthy or sick).

$$p(x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

the model found a **Sigmoid Function** that maps any real-valued input into an S-shaped curve bounded between 0 and 1, representing the probability.

<img width="1875" height="806" alt="image" src="https://github.com/user-attachments/assets/8e50a4be-fb59-465e-944c-3d94fbe4dd4d" />

