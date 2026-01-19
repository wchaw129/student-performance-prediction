# Student Exam Performance: Statistical Analysis & Predictive Modeling

This project focuses on predicting student academic performance based on socioeconomic and demographic factors. 

## 🎯 Project Objective
The primary goal was to build a robust model capable of predicting student scores. Instead of treating subjects (Math, Reading, Writing) in isolation, I performed a statistical analysis to determine if a consolidated "Average Score" could serve as a reliable, lower-dimensional target variable.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Analysis:** Pandas, NumPy, Statsmodels, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, CatBoost
* **Software Engineering:** Modularized code structure, `sklearn.pipeline`, Joblib

## 📊 Key Methodology & Statistical Insights

### 1. Statistical Target Consolidation (Notebook 01)
Before building the models, I conducted an in-depth statistical analysis to justify the pipeline architecture:
* **Target Consolidation & Correlation:** I performed a correlation analysis which revealed very strong positive relationships between subjects (e.g., Reading & Writing $r \approx 0.95$).
* **Normality Testing:** I conducted D’Agostino’s $K^2$ tests and visual inspections via Q-Q plots to verify whether the score distributions met the normality assumptions required for ANOVA.
* **ANOVA (Analysis of Variance):** I used ANOVA tests to determine if categorical features like `race/ethnicity` or `parental level of education` had a statistically significant impact on the mean scores. The results confirmed that these features are strong predictors, justifying their inclusion in the ML models.
* **Conclusion:** The statistical evidence (high correlation and similar distribution patterns) supported the use of the **Average Score** as a global indicator of academic performance, reducing target dimensionality without losing significant information.

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
 
### 4. Key findings
I compared two distinct modeling approaches:
* **The CatBoost model slightly outperformed the baseline OLS, capturing non-linear relationships between socioeconomic factors and student success.
* **Key performance drivers identified: Test Preparation Course and Lunch Type (socioeconomic proxy).


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
├── models/
│   └── student_performance_pipeline.joblib  # CatBoost model result
└── README.md
