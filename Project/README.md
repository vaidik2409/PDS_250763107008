# House Price Prediction

A machine learning project to predict house prices using the Ames Housing dataset.

---

## 📌 Overview

- **Dataset**: Ames Housing dataset (train and test CSV files included)
- **Goal**: Predict the `SalePrice` of residential homes
- **Techniques Used**:
  - Data cleaning and handling missing values
  - Feature engineering (combining area features, encoding categorical variables)
  - Training models like Ridge, Lasso, ElasticNet, and Gradient Boosting
  - Combining model predictions for better accuracy

---

## 📁 Files

- `house-price.ipynb` – Jupyter notebook with EDA, data preprocessing, and model training
- `pipeline.py` – Complete standalone Python script to run the pipeline
- `train.csv` & `test.csv` – Training and testing datasets
- `sample_submission.csv` – Sample submission format
- `submission.csv` – Final predicted prices
- `data_description.txt` – Description of all features in the dataset

---

## 🚀 How to Run

You can run the notebook directly:
```bash
jupyter notebook house-price.ipynb
```

Or run the Python script from terminal:
```bash
python pipeline.py
```
