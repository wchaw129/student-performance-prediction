# Student Exam Performance: Statistical Analysis & Predictive Modeling

This project focuses on predicting student academic performance based on socioeconomic and demographic factors. It demonstrates a full Data Science pipeline: from rigorous statistical verification and modular data engineering to advanced machine learning modeling.

## 🎯 Project Objective
The primary goal was to build a robust model capable of predicting student scores. Instead of treating subjects (Math, Reading, Writing) in isolation, I performed a statistical analysis to determine if a consolidated "Average Score" could serve as a reliable, lower-dimensional target variable.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Analysis:** Pandas, NumPy, Statsmodels, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, CatBoost
* **Software Engineering:** Modularized code structure, `sklearn.pipeline`, Joblib

## 📊 Key Methodology & Statistical Insights

### 1. Statistical Target Consolidation (Notebook 01)
Before modeling, I conducted a correlation analysis and statistical tests to verify the relationship between the three exam scores (Math, Reading, Writing).
* **Observation:** High positive correlation (e.g., Reading & Writing $r \approx 0.95$).
* **Statistical Decision:** Confirmed that modeling the **Mean Score** is statistically sound. This approach reduces noise, prevents redundant models, and provides a clearer picture of overall academic aptitude.

### 2. Feature Engineering & Pipelines
To ensure production-ready code, I implemented a custom preprocessing module (`src/preprocessing.py`) using `ColumnTransformer`:
* **Ordinal Encoding:** Applied to `parental level of education` to preserve the inherent hierarchy of degrees.
* **One-Hot Encoding:** Used for nominal features like `race/ethnicity`.
* **Binary Encoding:** Applied to features like `gender`, `lunch`, and `test preparation course`.
* **Pipeline Integration:** All steps are wrapped in a `scikit-learn Pipeline` to prevent data leakage and ensure reproducibility.

### 3. Predictive Modeling (Notebook 02)
I compared two distinct modeling approaches:
* **OLS (Ordinary Least Squares):** Utilized for baseline performance and to analyze feature significance (p-values) and coefficients, providing interpretability.
* **CatBoost Regressor:** Leveraged for its superior handling of categorical data and gradient boosting efficiency.
    * *Optimization:* Implemented Early Stopping to prevent overfitting.

## 📁 Project Structure
```text
├── notebooks/
│   ├── 01_exploring_data.ipynb  # EDA & Statistical verification
│   └── 02_models.ipynb          # Model training, comparison & evaluation
├── src/
│   ├── data_loader.py           # Data ingestion logic
│   ├── preprocessing.py         # Sklearn ColumnTransformer definition
│   └── __init__.py              # Package initialization
├── data/
│   └── StudentsPerformance.csv  # Raw dataset
└── README.md
