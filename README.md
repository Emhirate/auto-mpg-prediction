# Auto MPG Prediction: Machine Learning Regression

## Project Overview

This project develops a machine learning regression system to predict the fuel efficiency of automobiles, measured in miles per gallon (MPG), using vehicle characteristics such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin.

The project explores and compares multiple regression algorithms, applies a structured preprocessing workflow, engineers additional features, and evaluates model performance using cross-validation.

The primary objective is to identify a model that generalizes effectively to unseen data while demonstrating a reproducible and methodologically sound machine learning workflow.

## Problem Statement

Fuel efficiency is an important consideration in vehicle design, ownership, and environmental impact.

The objective of this project is to determine how vehicle characteristics influence fuel consumption and develop a regression model capable of predicting a vehicle's MPG from its specifications.

This is a supervised learning problem because the dataset contains known MPG values that serve as the target variable during model training.

## Dataset

The project uses the **Auto MPG dataset**, originally made available through the UCI Machine Learning Repository.

- **Dataset:** Auto MPG
- **Source:** [UCI Machine Learning Repository – Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg)
- **Task:** Regression
- **Target variable:** MPG
- **Observations:** 398 automobiles

### Dataset Features

| Feature      | Description                                           |
| ------------ | ----------------------------------------------------- |
| MPG          | Fuel efficiency in miles per gallon (target variable) |
| Cylinders    | Number of engine cylinders                            |
| Displacement | Engine displacement                                   |
| Horsepower   | Engine horsepower                                     |
| Weight       | Vehicle weight                                        |
| Acceleration | Time taken to accelerate                              |
| Model Year   | Year of vehicle manufacture                           |
| Origin       | Geographic origin of the vehicle                      |
| Car Name     | Vehicle identification/name                           |

The `Car Name` field is not used as a predictive feature in the modelling workflow.

The dataset contains missing values in the `Horsepower` feature, which are addressed through the preprocessing pipeline.

## Project Workflow

The project follows a structured machine learning workflow.

### 1. Data Exploration and Understanding

Initial exploratory data analysis (EDA) is performed to understand the dataset's structure, identify missing values, examine feature distributions, and investigate relationships between vehicle characteristics and fuel efficiency.

### 2. Data Preparation

The dataset is prepared for modelling through:

- Identification and handling of missing values.
- Separation of predictors and target variable.
- Encoding of categorical variables.
- Separation of training and testing datasets.

The test dataset is kept separate from model development to support an unbiased final evaluation.

### 3. Feature Engineering

A custom transformer is implemented to generate additional predictive features:

- `acceleration_on_cyl`: Acceleration divided by the number of cylinders.
- `acceleration_on_power`: Acceleration divided by horsepower.

These engineered features are incorporated into the numerical preprocessing pipeline.

### 4. Data Preprocessing

A Scikit-learn preprocessing pipeline is constructed to ensure that transformations are applied consistently.

The preprocessing workflow includes:

- **Missing-value imputation:** Median imputation for numerical features.
- **Feature engineering:** Generation of additional numerical attributes.
- **Feature scaling:** Standardization using `StandardScaler`.
- **Categorical encoding:** One-hot encoding of the `Origin` feature.

The preprocessing transformations are integrated with each estimator in a complete Scikit-learn pipeline.

This ensures that preprocessing is fitted independently within each cross-validation training fold, preventing data leakage.

### 5. Model Development

Four regression algorithms are evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression (SVR)

Each model is evaluated using the same preprocessing approach to ensure a fair comparison.

### 6. Model Evaluation

Ten-fold cross-validation is used to assess model performance.

The primary evaluation metric is Root Mean Squared Error (RMSE).

RMSE measures the typical magnitude of prediction errors, with larger errors receiving greater penalties.

A lower RMSE indicates better predictive performance.

The standard deviation of cross-validation RMSE scores is also examined to understand the variability of model performance across folds.

## Baseline Model Comparison

The following results were obtained during the initial ten-fold cross-validation experiment:

| Model                     | Mean CV RMSE | CV RMSE Standard Deviation |
| ------------------------- | -----------: | -------------------------: |
| Random Forest             |   **2.6029** |                     0.5838 |
| Linear Regression         |       3.0742 |                     0.4849 |
| Support Vector Regression |       3.0857 |                     0.4864 |
| Decision Tree             |       3.1141 |                     0.5327 |

### Initial Findings

The Random Forest Regressor achieved the lowest mean cross-validation RMSE of approximately **2.60 MPG**, making it the strongest-performing baseline model among the four algorithms evaluated.

Linear Regression and Support Vector Regression produced relatively similar results, while the Decision Tree Regressor recorded the highest mean cross-validation RMSE.

These findings support selecting Random Forest as the leading candidate for further hyperparameter optimization.

*Note: These are baseline cross-validation results, not final test-set performance or confirmed post-tuning results.*

## Hyperparameter Optimization

Random Forest is selected for further investigation using `GridSearchCV`.

The optimization process explores different combinations of:

- Number of trees.
- Number of features considered at each split.
- Bootstrap sampling (`bootstrap`).

The objective is to identify a configuration that improves predictive performance while maintaining reasonable generalization.

The final selected configuration should be determined from cross-validation results rather than training-set performance.

## Technologies and Libraries

The project uses Python and the following libraries:

| Technology       | Purpose                                          |
| ---------------- | ------------------------------------------------ |
| Python           | Primary programming language                     |
| Jupyter Notebook | Interactive development and documentation        |
| Pandas           | Data manipulation and analysis                   |
| NumPy            | Numerical computation                            |
| Matplotlib       | Data visualization                               |
| Seaborn          | Statistical data visualization                   |
| Scikit-learn     | Preprocessing, model development, and evaluation |


## Reproducibility

The project uses fixed random states where applicable to improve reproducibility.

The preprocessing and modelling operations are organized into scikit-learn pipelines to ensure consistent transformations during training, cross-validation, hyperparameter optimization, and prediction.

## Limitations

Some limitations of the project include:

- The dataset is relatively small compared with modern automotive datasets.
- The dataset represents older vehicle models and may not reflect the characteristics of contemporary automobiles.
- Model performance depends on the quality and representativeness of the available features.
- Cross-validation performance does not guarantee identical performance on entirely new datasets.

## Future Improvements

Potential extensions include:

- Exploring additional regression algorithms.
- Investigating alternative feature-engineering strategies.
- Comparing different hyperparameter optimization approaches.
- Analyzing feature importance and model interpretability.
- Evaluating the final selected model on the untouched test set.
- Investigating model performance across different vehicle categories.

## Author

**Azeez Idris**

Machine Learning and Data Science Portfolio Project

---

## Acknowledgements

The Auto MPG dataset is provided by the UCI Machine Learning Repository.

Dataset reference: [Auto MPG – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg)
