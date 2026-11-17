# Auto MPG Prediction

A machine learning regression project that predicts the fuel efficiency of vehicles using their technical specifications. The project explores data analysis, preprocessing, feature engineering, model development, cross-validation, and hyperparameter optimisation.

## Project Overview

Fuel efficiency is an important consideration in vehicle analysis. This project uses the Auto MPG dataset to investigate how vehicle characteristics relate to fuel consumption and develops regression models to predict miles per gallon (MPG).

The primary objective is to build and evaluate machine learning models, identify the most effective approach, and examine whether hyperparameter optimisation improves predictive performance.

## Dataset

The project uses the **Auto MPG dataset** from the UCI Machine Learning Repository.

- **Observations:** 398 vehicles
- **Target variable:** MPG
- **Features:** Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, and Origin.
- **Missing values:** 6 missing observations in Horsepower.

The target variable, MPG, represents a vehicle's fuel efficiency in miles per gallon.

## Project Workflow

### 1. Data Exploration and Analysis

The dataset was examined to understand its structure, identify missing values, investigate feature distributions, and explore relationships between vehicle characteristics and fuel efficiency.

### 2. Data Preprocessing

The preprocessing workflow includes:

- Handling missing numerical values using median imputation.
- Encoding categorical vehicle-origin information.
- Scaling numerical features using StandardScaler.
- Using scikit-learn pipelines to organise preprocessing and model training.

### 3. Feature Engineering

Two additional features were developed:

- `acceleration_on_cyl`
- `acceleration_on_power`

These features explore relationships between acceleration, engine cylinders, and horsepower.

### 4. Model Development

Four regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)

Random Forest achieved the strongest baseline cross-validation performance and was selected for further optimisation.

### 5. Hyperparameter Optimisation

Random Forest hyperparameters were investigated using `GridSearchCV` with 10-fold cross-validation.

The expanded search evaluated 18 parameter combinations, resulting in 180 model fits.

The search investigated:

| Hyperparameter | Values evaluated |
|---|---|
| `n_estimators` | 100, 200 |
| `max_features` | 0.5, 0.75, 1.0 |
| `min_samples_leaf` | 1, 2, 4 |

The selected configuration was:

| Hyperparameter | Selected value |
|---|---:|
| `n_estimators` | 200 |
| `max_features` | 1.0 |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |

### 6. Cross-Validation Results

The baseline and tuned Random Forest models were compared using 10-fold cross-validation.

| Metric | Baseline Random Forest | Tuned Random Forest |
|---|---:|---:|
| Mean CV RMSE | 2.602 MPG | 2.591 MPG |
| Standard deviation | 0.583 | 0.600 |

The tuned model achieved an approximate 0.42% reduction in mean cross-validation RMSE.

However, the improvement was marginal and was not consistent across all validation folds. The tuned model also showed slightly higher fold-to-fold variability.

Therefore, hyperparameter optimisation produced a small improvement rather than a substantial increase in predictive performance.

The tuned Random Forest was retained as the selected model.

*Note: These are cross-validation results, not final test-set performance.*

## Technologies and Libraries

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
auto-mpg-prediction/
│
├── data/
│   └── auto-mpg.data
│
├── notebooks/
│   └── mpg_pro.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Emhirate/auto-mpg-prediction.git
cd auto-mpg-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open the project notebook:

```bash
jupyter notebook
```

Navigate to `notebooks/mpg_pro.ipynb` and execute the cells sequentially.

Ensure the dataset is available in the expected `data/` directory.

## Reproducibility

The project uses a fixed random state (`random_state=42`) for reproducibility in the relevant data-splitting and model-building operations.

Preprocessing is incorporated into machine learning pipelines to ensure that transformations are fitted within the training folds during cross-validation.

## Limitations

- The dataset contains only 398 observations.
- The dataset represents older vehicle models and may not reflect the characteristics of modern vehicles.
- The improvement from hyperparameter optimisation was marginal.
- Cross-validation performance should not be interpreted as a guarantee of performance on unseen data.

## Future Improvements

- Add model evaluation visualisations.
- Examine actual versus predicted MPG values and residuals.
- Improve model interpretation and feature-importance analysis.
- Explore additional regression algorithms where justified.

## Author

**Azeez Idris**

GitHub: [@Emhirate](https://github.com/Emhirate)

---

*This project was developed as a practical demonstration of an end-to-end machine learning regression workflow.*
