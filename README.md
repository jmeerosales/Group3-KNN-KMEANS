# Group3-KNN-KMEANS

# Diabetes Analysis using KNN 

This project explores the application of **Supervised (KNN)** machine learning algorithms to analyze and predict diabetes outcomes.

## Project Goal
The objective is to accurately classify patients as diabetic or non-diabetic using the K-Nearest Neighbors algorithm.

## Data Preprocessing (The Workflow)
A critical part of this project was preparing the clinical data for distance-based algorithms:

1. **Median Imputation**: Replaced invalid "0" values in features like Glucose, Blood Pressure, and Insulin with the median to maintain medical accuracy and prevent data bias.
2. **Z-Score Standardization**: Scaled all features to a mean of 0 and standard deviation of 1. This ensures that high-value features (like Insulin) don't overpower smaller-value features (like BMI) during distance calculations.

## Algorithms Used

### 1. K-Nearest Neighbors (KNN)
* **Type**: Supervised Learning (Classification)
* **Method**: Predicts a label by looking at the $K$ most similar neighbors in the multi-dimensional feature space.

## Conclusion

The implementation of the **KNN algorithm** on this dataset demonstrates that predictive accuracy in medical diagnostics relies heavily on meticulous data preprocessing. By addressing "hidden" missing values (zeros) through **Median Imputation**, we preserved the dataset's integrity while ensuring biological plausibility.

The application of **Z-Score Standardization** was the most critical step; without it, the distance-based logic of KNN would have been skewed by the varying scales of clinical measurements. With an optimized value of **$K=19$**, the model achieves a reliable balance between sensitivity and specificity, proving that when data is cleaned and scaled correctly, simple algorithms can provide powerful diagnostic insights.

---
