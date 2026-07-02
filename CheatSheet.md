----

# `Scaling`

| Situation                                                 | Recommended Scaler     |
| --------------------------------------------------------- | ---------------------- |
| Approximately normal distribution                         | StandardScaler         |
| Need values between 0 and 1                               | MinMaxScaler           |
| Deep learning/image pixels                                | MinMaxScaler           |
| PCA                                                       | StandardScaler         |
| KNN, K-Means, SVM                                         | Usually StandardScaler |
| Linear/Logistic Regression                                | StandardScaler         |
| Presence of many outliers                                 | RobustScaler           |
| Tree-based models (Random Forest, XGBoost, Decision Tree) | No scaling needed      |

---

# `Skewness`

| Skewness Value | Type                       | Transformation                       |
| -------------- | -------------------------- | ------------------------------------ |
| -0.5 to 0.5    | Approximately symmetric    | No transformation                    |
| 0.5 to 1       | Mild right skew            | √x or log1p                          |
| > 1            | Moderate/Severe right skew | log1p or PowerTransformer            |
| -1 to -0.5     | Mild left skew             | x²                                   |
| < -1           | Moderate/Severe left skew  | Reflection + log or PowerTransformer |
