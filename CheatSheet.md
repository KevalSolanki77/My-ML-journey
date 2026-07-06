----

# `Data Cleaning`

Handaling the datetime Columns,<br>
Handling the object Columns

---

#  `EDA - Visualization`

## 1. Univariate Analysis: Numerical Columns

When analyzing a single numerical column, the goal is to understand its central tendency, spread, shape of distribution, and the presence of outliers.

### Key Metrics to Compute
* **Central Tendency:** Mean, Median, Mode.
* **Spread/Dispersion:** Variance, Standard Deviation, Range, Interquartile Range (IQR).
* **Shape:** Skewness (asymmetry) and Kurtosis (tailedness).

### Visualizations
* **Histogram:** To see the distribution shape (Normal, skewed, bimodal).
* **Kernel Density Estimate (KDE) Plot:** A smooth version of the histogram.
* **Box Plot (Whisker Plot):** To clearly identify outliers, median, and quartiles.

## 2. Univariate Analysis: Categorial Columns

When analyzing a single categorical column, the focus shifts to frequency distribution, cardinality (number of unique values), and balancing issues.

### Key Metrics to Compute
* **Frequency Count:** Number of occurrences for each unique category.
* **Relative Frequency:** Percentage share of each category.
* **Cardinality:** Total number of unique categories (df['col'].nunique()).

### Visualizations
* **Bar Chart:** To compare the frequency or percentage of each category.
* **Pie Chart:** Useful only if categories are few (less than 5) to show parts-of-a-whole.

## 3. Bivariate & Multivariate Analysis (Combining Both)

To find relationships, patterns, and dependencies across different types of features, we combine columns using specific techniques based on their data type mixes.

### Mix A: Numerical vs. Numerical

* **Objective:** Find correlation, linearity, and joint distributions.
* **Metrics:** Pearson Correlation (linear), Spearman Rank Correlation (non-linear/monotone).
* **Visualizations:** Scatter Plot, Pair Plot (for multi-variable grids), Heatmap (for correlation matrix).

### Mix B: Categorical vs. Categorical

* **Objective:** Find dependencies or associations between distinct groupings.
* **Metrics:** Chi-Square Test of Independence, Cramér's V.
* **Visualizations:** Stacked Bar Chart, Grouped Bar Chart, Crosstab Heatmap.

### Mix C: Numerical vs. Categorical

* **Objective:** Compare numerical values across different groups/categories.
* **Metrics:** Grouped statistical summaries (Mean/Median per category), ANOVA (parametric test), Kruskal-Wallis (non-parametric).
* **Visualizations:** Box Plot grouped by category, Violin Plot, Facet Grid of Histograms.

# `Encoding`

## 1. Label Encoding

* **Used:** for Target Column if the target Columns is Categorical
```python
from sklearn.preprecessing import LabelEncoder
encoder = LabelEncoder()
```

## 2. Ordinal Encoding

* **Used:** when the Column's Categories have `Natural Order` means one category is superior to others.
* **For example:** `car_owner` has categories like `First Owner, Second_owner, Third Owner, etc`. 
```python
from sklearn.preprecessing import OrdinalEncoder
encoder = OrdinalEncoder(categories = [['First Owner', 'Second_owner', 'Third Owner']], )
```

## 3. One Hot Encoding
* **Used:** When the Columns have `no order` means all the categories are treated equally.
* **For example:** `gender` has no order.
```python
from sklearn.preprecessing import OneHotEncoder
encoder = OneHotEncoder(dtype = int, drop = 'first', sparse_output = False)
```

# `Imputation`

## Missingness Checking

1. Run Little's MCAR test 
```python 
from pyamute.exploration.mcar_statstical_tests import MCARTest
```

   ├── p > 0.05 → Treat as MCAR → safe to use mean/median/mode or even drop rows

   └── p < 0.05 → Not MCAR → go to step 2

2. Test missingness indicator against other observed features
   (chi-square / logistic regression / groupby comparison)

   ├── Strong relationship found → MAR → use model-based imputation 

   │   (IterativeImputer/KNNImputer — conditions on other features)

   └── No relationship found, but you also can't explain missingness 
       by anything observed → Suspect MNAR

3. For suspected MNAR → think about domain knowledge:
   - Add a "missingness flag" as its own feature (often the most honest move)
   - Consider Heckman-style selection models in production-grade work
   - At minimum, document the assumption — don't pretend it's MCAR

# `Outliers Handling`

## Statistical methods:

### IQR Method (most common, robust to skew)

* Q1 = 25th percentile
* Q3 = 75th percentile
* IQR = Q3 − Q1
* Lower bound = Q1 − 1.5×IQR
* Upper bound = Q3 + 1.5×IQR
```python 
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
IQR = q3 - q1
lower = q1 - 1.5*IQR
upper = q3 - 1.5*IQR
outliers = df[df[col] < Lower Bound] | df[df[col] > Upper Bound]]
```

Z-score method (best for normally distributed data)

z = (x − mean) / std
|z| > 3 is typically flagged as outlier
Weakness: mean & std themselves are sensitive to outliers, so this can be misleading on skewed data


Modified Z-score (using Median Absolute Deviation) — more robust version of Z-score, good when data isn't normal

Algorithmic / ML-based (for multivariate or high-dimensional outliers):

Isolation Forest — isolates anomalies instead of profiling normal points; very fast, scales well
Local Outlier Factor (LOF) — flags points that are in low-density regions relative to neighbors
DBSCAN — clustering algorithm that naturally labels sparse points as noise

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

---

# `Missingness Checking`

1. Run Little's MCAR test 
```python 
from pyamute.exploration.mcar_statstical_tests import MCARTest
```

   ├── p > 0.05 → Treat as MCAR → safe to use mean/median/mode or even drop rows

   └── p < 0.05 → Not MCAR → go to step 2

2. Test missingness indicator against other observed features
   (chi-square / logistic regression / groupby comparison)

   ├── Strong relationship found → MAR → use model-based imputation 

   │   (IterativeImputer/KNNImputer — conditions on other features)

   └── No relationship found, but you also can't explain missingness 
       by anything observed → Suspect MNAR

3. For suspected MNAR → think about domain knowledge:
   - Add a "missingness flag" as its own feature (often the most honest move)
   - Consider Heckman-style selection models in production-grade work
   - At minimum, document the assumption — don't pretend it's MCAR
