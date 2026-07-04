# 🚀 My Machine Learning Journey

Documenting my day-by-day progress as I learn ML — concepts, code, and mistakes along the way.

## 📌 About
- **Goal:** [e.g. Build a strong foundation in ML and land an ML role]
- **Started:** [Date]
- **Focus areas:** Data preprocessing, ML algorithms, math foundations

## 🗂️ Progress Tracker

| Day | Topic | Status |
|-----|-------|--------|
| 1 | NumPy | ✅ |
| 2 | Pandas | ✅ |
| 3 | Data Collection | ✅ |
| 4 | Data Visualization | ✅ |
| 5 | Imputation | ✅ |
| 6 | Encoding | ✅ |
| 7 | Feature Scaling | ✅ |
| 8 | Column Transformer | ✅ |
| 9 | Function Transformer | ✅ |
| 10 | Pipelines | ✅ |
| 11 | Binning & Binarization | ✅ |
| 12 | Linear Regression (OLS) | ✅ |
| 13 | Gradient Descent | ✅ |
| 14 | Polynomial Regression | ✅ |
| 15 | Condition Number & VIF | ✅ |

## 📖 Daily Logs

### Day 1: NumPy
**What I learned:**
- Creating arrays with `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, and `np.linspace()`
- Difference between 1D, 2D, and 3D arrays, and how `.reshape()` and `.ravel()` work
- Indexing and slicing 2D arrays, including negative step slicing and boolean masking (`arr[arr > 40]`)
- Vectorized operations (`+`, `*`, `**`) and the dot product (`@`) vs matrix multiplication
- Broadcasting: how NumPy handles operations between arrays of different shapes
- Aggregate functions (`sum`, `mean`, `std`, `max`, `min`) with `axis` parameter for row/column-wise stats
- Linear algebra with `np.linalg`: determinant, inverse, transpose, and eigen decomposition
- Reconstructing a matrix from its eigenvectors and eigenvalues (`V @ diag(λ) @ V.T`) and verifying it with `np.allclose()`
- Generating random data with `np.random.seed()`, `randint()`, `rand()`, and `normal()` for reproducibility
- Row-wise normalization (z-score) using `axis=1` with `keepdims=True`
- Cleaning sentinel values (e.g., `-999` as missing data) using boolean filtering
- Stacking and concatenating arrays with `np.hstack()` and `np.concatenate()`
- Splitting arrays with `np.split()`
- Conditional replacement with `np.where()` and in-place updates with `np.put()`
  
**Code:** [NumPy.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-01-Numpy/NumPy.ipynb)

---

### Day 2: Pandas
**What I learned:**
- Creating and indexing `pd.Series` with custom labels, and the difference between `.loc` (label-based) and `.iloc` (position-based)
- Boolean filtering on Series with chained conditions using `&`/`|` (not `and`/`or`, which pandas doesn't support)
- Building a `pd.Series` directly from a dictionary
- Loading a CSV into a DataFrame with `pd.read_csv()`, selecting specific columns via `usecols`
- Exploring a DataFrame: `.head()`, `.tail()`, `.columns`, `.shape`, `.isnull().sum()`, `.unique()`, `.value_counts()`
- Filtering rows with boolean conditions (`df[df.col > x]`) and combining with `.loc`
- Handling missing values: `.fillna()` with a constant, mean, or median, and `.dropna()`
- Sorting with `.sort_values()`, including multi-column sort and `ascending=False`
- Grouping and aggregating with `.groupby()` — mean, sum, size, `.agg()` with multiple functions
- Creating new columns from existing ones (feature engineering), including boolean-to-int conversion with `.astype(int)`
- Combining `groupby()` with `.head()` to get top-N rows per group
- Using `.idxmax()` / `.idxmin()` to find the row/group with the max or min value
- Computing correlation between numeric columns with `.corr()`
- Finding duplicate rows with `.duplicated()`
- Converting columns to numeric safely with `pd.to_numeric(..., errors='coerce')`
- Min-max normalization of numeric columns using `.select_dtypes()` to target only numeric types
- Reshaping data with `.pivot_table()`

**Code:** [pandas.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-02-Pandas/pd.ipynb)

---

### Day 3: Data Collection
**What I learned:**
- Reading existing datasets with `pd.read_csv()` and selecting specific columns
- Calling a REST API with the `requests` library, including passing custom headers and query parameters (e.g. RapidAPI setup)
- Web scraping with `BeautifulSoup` + `requests`: fetching HTML, parsing it, and extracting elements by tag and CSS class
- Writing a custom parser function to convert messy string formats (e.g. `"1.2L"`, `"500K"`, `"2CR"`) into clean integers
- Scraping data across multiple paginated pages using a loop
- Structuring scraped lists into a clean `pd.DataFrame`
- Doing an initial data audit on a freshly collected dataset: `.shape`, `.head()`, `.info()`, `.isnull().sum()`, `.duplicated().sum()`
- Checking correlation between a target column and other numeric features with `.corr()`
- Visualizing relationships with `seaborn.scatterplot()`
- Quickly testing whether a feature has predictive value by fitting a baseline `RandomForestRegressor` and checking `.score()`

**Code:** [data_collection.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-03-Data-Collection/data_collections.ipynb)

---

### Day 4: Data Visualization
**What I learned:**
- Line, bar, grouped bar, histogram, and scatter plots with Matplotlib
- Customizing charts: titles, labels, grid, legend, colors, and annotations (`plt.text`)
- Conditional bar coloring (e.g. green/red for profit vs loss)
- Grouped bar charts using offset x-positions for multi-category comparison
- Quick styled scatter plots with Seaborn (`sns.scatterplot`)

**Code:** [visualization.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-04-Visualization/visualization.ipynb)

---

### Day 5: Imputation
**What I learned:**
- Real world data is so much messy and some of themm might be empty to fill this we do Imputation
- With `SimpleImputer()` we can impute data by `strategy=mean/median` for Numerical data by the mean/median of data
- With `strategy=most_frequent` we can impute categorical data by Mode of the data
- Another technic is `Random Sampling Imputation` but there is no direct method in the `sklearn` to implement this.
- In this imputation we drop the rows with Nan value and take random sample from the remain data.
- Benefit of this imputation is that it maintains the distribution of the data.
- With `IterativeImputer()` we can predict columns that have very high correlation.
- `IterativeImputer()` uses `RandomForestRegressor()` as the `base_estimator` that predicts the columns as target value using other columns as features in a loop of `max_iter`.
- `max_iter` is a hyperparameter, so if `max_iter` is low then it won't get the perfect results.
- `KNNImputer()` uses the neighbour to calculate the value of missing row.
- `n_neighbours` is the hyperparameter. 

**Code:** [imputer.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-05-Imputation/imputer.ipynb)

---

### Day 6: Encoding
**What I learned:**
- For input Features There are two types of Categorial Data: 1) `Ordinal` 2) `Nominal` 
- `Ordinal` means there is a Natural Order of categories and `Nominal` means there is no order all the values are treated equally.
- Based on these categories There are two Types of Encoders
  1. `OrdinalEncoder()` - for `Ordinal` Categories
  2. `OneHotEncoder()` - for `Nominal` Categories
- In `OrdinalEncoder()` you define the order using `categories=` parameter. By default it will assign the value starting from `0...n`. `0` means lowest and `n` means Highest order.
- By doing this Model will give more value to the higher order and make more precise predictions.
- In `OneHotEncoder()` we'll create a new column for every category and set the value `0` or `1`.
- But problem with `OneHotEncoder()` is that if we create column for every category that make aa those columns very Highly Correlated because sum of those encoded columns will be 1 because there is 1 category per row. To solve these there is a parameter called `drop` we can set the value to `first` that will drop the first column from encoded column the encoded columns.
- We can also achieve same outputs by using `pandas.get_dummies()`.
- For Target features we have `LabelEncoder()` that encodes the target features.

**Code:** [encoding.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-06-Encoding/encoding.ipynb)


---

### Day 7: Feature Scaling
**What I learned:**
- Feature scaling matters because features on different ranges (like height in cm vs weight in kg) can bias models that rely on distance or gradients — the feature with a bigger range ends up dominating.
- There are mainly two types of scaling: `Standardization` and `Normalization`.
- `StandardScaler()` performs standardization — it transforms data to have mean `0` and standard deviation `1` using the formula `(x - mean) / std`. It doesn't bound values to a fixed range, so outliers can still stretch the distribution.
- `MinMaxScaler()` performs normalization — it squeezes all values into a fixed `[0, 1]` range using `(x - min) / (max - min)`. This is useful when you need bounded values, but it's very sensitive to outliers since a single extreme value shifts the entire scale.
- `RobustScaler()` is built for data with outliers. Instead of mean/std or min/max, it scales using the median and IQR (Interquartile Range): `(x - median) / IQR`. Since median and IQR aren't pulled by extreme values the way mean and range are, the scaled distribution stays stable even with outliers present.
- `MaxAbsScaler()` scales data by dividing by the maximum absolute value, so everything lands in `[-1, 1]`. It doesn't shift/center the data (no subtraction), so it preserves sparsity — this is useful for sparse data where you don't want to disturb the zero entries.
- Choosing a scaler depends on the data: use `StandardScaler` when data is roughly normal, `MinMaxScaler` when you need a strict bounded range and there are no major outliers, `RobustScaler` when outliers are present, and `MaxAbsScaler` when working with sparse data.
- Visualizing the same feature's distribution before and after scaling (using `histplot` with KDE) helps confirm that scaling changes the *scale* of the data but not its underlying *shape*.


**Code:** [feature_scaling.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-07-Feature-scaling/feature_scaling.ipynb)

---

### Day 8: Column Transformer
**What I learned:**
- 

---

### Day 9: Function Transformer
**What I learned:**
- 

---

### Day 10: Pipelines
**What I learned:**
- 

---

### Day 11: Binning & Binarization
**What I learned:**
- 

---

### Day 12: Linear Regression (OLS)
**What I learned:**
- 

---

### Day 13: Gradient Descent
**What I learned:**
- 

---

### Day 14: Polynomial Regression
**What I learned:**
- 

---

### Day 15: Condition Number & VIF
**What I learned:**
- 

---

## 🛠️ Tech Stack
`Python` `NumPy` `Pandas` `Matplotlib` `Seaborn` `Scikit-learn`

## 🔗 Connect
- LinkedIn: [your profile]
- GitHub: [your profile]
