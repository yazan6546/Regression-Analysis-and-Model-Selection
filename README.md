# Regression-Analysis-and-Model-Selection

This repository contains code and data for comparing various regression models to predict car prices using a dataset from the YallaMotor website. The dataset includes multiple features such as car name, price, engine capacity, cylinder count, horsepower, top speed, number of seats, brand, and country of sale. The goal is to evaluate the effectiveness of different regression techniques in predicting car prices and performance metrics.

## Table of Contents

 - [Introduction](#introduction)
 - [Dataset](#dataset)
 - [Preprocessing](#preprocessing)
   - [Cleaning](#cleaning)
   - [Encoding](#encoding)
   - [Scaling](#scaling)
 - [Models](#models)
 - [Results](#results)
 - [Conclusion](#conclusion)
 - [Repository Structure]
(#repository-structure)
 - [Usage](#usage)
 - [Dependencies](#dependencies)
 - [Technical Report](#technical-report)
 - [Contributors](#contributors)

## Introduction

This project aims to compare various regression models to predict car prices using a comprehensive dataset from the YallaMotor website. The analysis evaluates the performance of different regression techniques and identifies the most suitable models for accurate car price prediction.

## Dataset

The dataset contains the following columns:

- **Car Name**: The name of the car.
- **Price**: The price of the car.
- **Engine Capacity**: The car's engine capacity.
- **Cylinder**: The car's cylinder count.
- **Horse Power**: The car's horsepower.
- **Top Speed**: The car's top speed.
- **Seats**: The number of seats in the car.
- **Brand**: The car's brand.
- **Country**: The country where the site sells this car.

## Preprocessing

The dataset underwent several preprocessing steps to ensure data quality and suitability for modeling. The preprocessing is divided into three main subsections: Cleaning, Encoding, and Scaling.

### Cleaning

Data cleaning involved handling missing values, correcting erroneous entries, and standardizing data formats.

1. **Dropped Duplicate Rows**: Removed any duplicate entries to ensure each record is unique.

2. **Handled Non-Numeric Values in `horse_power`**:
   - Replaced all non-numeric values with `NaN`.
   - Dropped all missing values in `horse_power`.

3. **Handled Non-Numeric Values in `top_speed`**:
   - Replaced all non-numeric values with `NaN`.

4. **Removed Whitespace**: Stripped leading and trailing whitespaces from all columns.

5. **Converted Currencies to USD**: Standardized all price values to USD for consistency.

6. **Handled Non-Numeric Values in `price`**:
   - Replaced non-numeric entries with `NaN`.
   - Used mean imputation grouped by `car_name` to fill missing values.

7. **Handled Remaining Null Values in `price`**:
   - Used median imputation grouped by `brand` for any remaining missing values.

8. **Corrected `top_speed` Entries**:
   - Exchanged erroneous `top_speed` values that contained seater information.

9. **Mode Imputation for Inconsistent Entries**:
   - Used mode imputation grouped by `car_name` to correct inconsistent entries due to data entry errors.

10. **Handled `N A` Values in `seats`**:
    - Replaced `N A` entries with `NaN`.
    - Used median imputation grouped by `brand` to fill missing values.

11. **Extracted Numeric Values from `seats`**:
    - Extracted numerical data from entries containing the word "Seater".

12. **Handled `N/A, Electric` in `cylinder`**:
    - Replaced `N/A, Electric` values with `0`.

13. **Inferred Missing `cylinder` Values**:
    - Inferred missing values based on `engine_capacity`.

14. **Standardized `engine_capacity` Units**:
    - Converted values with different units to liters.

15. **Handled Outliers in `horse_power` and `top_speed`**:
    - Used median imputation after grouping by `brand` to handle outliers.

16. **Extracted `year` from `car_name`**:
    - Extracted the manufacturing year and subsequently dropped the `car_name` column.

### Encoding

Encoding was applied to transform categorical variables into numerical formats suitable for modeling.

1. **One-Hot Encoding for `country`**:
   - Applied one-hot encoding to the `country` column.
   - Transformed categorical country names into binary columns representing each country.

2. **Target Encoding for `price`**:
   - Applied target encoding to the `price` column where appropriate.
   - Replaced categories with the mean of the target variable to reduce dimensionality.

### Scaling

Scaling was performed to normalize the features and bring them to a common scale.

1. **Min-Max Scaling**:
   - Applied Min-Max scaling to all numerical features.
   - Transformed features to a fixed range of [0, 1] using the formula:

     \[
     X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
     \]

     where:

     - \(X\) is the original feature value.
     - \(X_{\text{min}}\) and \(X_{\text{max}}\) are the minimum and maximum values of the feature.
     - \(X'\) is the scaled feature value.

## Models

The following regression models were compared:

- **Linear Regression**
- **Lasso Regression**
- **Ridge Regression**
- **Support Vector Regression (SVR)**

## Results

The results of the regression models were evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) score. The models were compared to determine the most effective regression technique for predicting car prices.

## Conclusion

The analysis demonstrated that certain regression models outperformed others in predicting car prices. The results indicate that appropriate preprocessing and the choice of regression model are crucial for achieving accurate predictions. This comparison provides valuable insights into the strengths of different regression techniques when applied to complex datasets in the automotive domain.

## Repository Structure

The repository has the following structure:

    car-price-prediction/
    ├── data/
        └── data.csv
        └── cleaned_cars.csv
    ├── report/
        └── Technical_report.pdf
    ├── cleaning.ipynb
    ├── main.ipynb
    ├── regression.py
    ├── requirements.txt
    ├── README.md

- **`data/cleaned_cars.csv`**: The cleaned dataset generated by `cleaning.ipynb`, used for training and evaluating the regression models.
- **`report/Technical_report.pdf`**: A technical report containing a comprehensive discussion and analysis in greater detail.
- **`data/data.csv`**: The original dataset. This file can be generated by running the first cell in `cleaning.ipynb`.
- **`cleaning.ipynb`**: A Jupyter Notebook that contains the data cleaning process. It reads the raw dataset, applies preprocessing steps, and saves the cleaned dataset.
- **`main.ipynb`**: A Jupyter Notebook that illustrates the implementation and comparison of various regression models. It loads the cleaned dataset and evaluates different regression techniques.
- **`regression.py`**: A Python module containing custom regression functions used in `main.ipynb`.
- **`requirements.txt`**: A file listing the Python dependencies required to run the code.
- **`README.md`**: The README file providing an overview of the project.
- **`LICENSE`**: The license file for the project.

## Usage

To run the code in this repository, follow these steps:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/yazan6546/Regression-Analysis-and-Model-Selection.git
   cd car-price-prediction
   ```
2. **Navigate to its directory**:
   ```sh
   cd Regression-Analysis-and-Model-Selection
   ```
2. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the data cleaning notebook:
   ```sh
   jupyter notebook cleaning.ipynb
   ```
   Or open using your favorite IDE such as vscode.

   - This notebook will clean the raw dataset and save the cleaned data to `cleaned_cars.csv`.
4. Run the main analysis notebook:
   ```sh
   jupyter notebook main.ipynb
   ```
   - This notebook will load the cleaned dataset, implement various regression models, and display the results.
     
## Dependencies

The following Python libraries are required to run the code:

- pandas
- numpy
- scikit-learn
- category_encoders
- matplotlib
- IPython
- Jupyter Notebook

## Technical Report

A detailed technical report is available in this repository, providing an in-depth analysis of the data preprocessing steps, model selection, and evaluation metrics. The report includes visualizations and explanations of the results, offering valuable insights into the regression analysis and model selection process. You can check the technical report [here](report/Technical_report.pdf)..


## Contributors 

- [Yazan AbuAloun](https://github.com/yazan6546)
- [Ahamd Qaimari](https://github.com/ahmadq44)

   

