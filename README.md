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
- Real datasets often need different preprocessing for different columns at the same time — one column might need imputation, another ordinal encoding, another one-hot encoding. Doing each of these manually with separate `fit_transform` calls gets messy and error-prone, especially when applying the same transformations consistently to both train and test sets.
- `ColumnTransformer` solves this by letting you define all these column-specific transformations in one place, as a list of `(name, transformer, columns)` tuples, and applying them together in a single `fit_transform()` call.
- Each transformer inside `ColumnTransformer` only sees the columns assigned to it — so `SimpleImputer` only touches the numeric column, `OrdinalEncoder` only touches the ordinal column, and `OneHotEncoder` only touches the nominal columns, all in parallel.
- The `remainder='passthrough'` parameter tells `ColumnTransformer` to keep any columns not explicitly listed, instead of dropping them — otherwise only the transformed columns would survive.
- `verbose_feature_names_out=False` keeps the output column names clean (e.g. `age`, `cough`) instead of prefixing them with the transformer name (e.g. `imputer__age`).
- Since `ColumnTransformer` outputs a NumPy array by default, wrapping it back into a `pd.DataFrame` using `.get_feature_names_out()` for column names keeps the data readable after transformation.
- The key benefit: once fitted on `X_train`, the exact same `transformer` object can be applied to `X_test` using `.transform()` — guaranteeing train and test data go through identical preprocessing, avoiding data leakage or inconsistent encoding between the two sets.

**Code:** [column_transformer.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-08-Column-Transformer/column_transformer.ipynb)

---

### Day 9: Function Transformer
**What I learned:**
- Many features aren't naturally normal — they're skewed, and models like Logistic/Linear Regression assume roughly normal input, so skewed features can hurt performance. Checking `.skew()` per column shows which features need this treatment.
- `FunctionTransformer` lets you apply any custom or NumPy function (like `np.log1p` or `np.square`) as a transformation step inside a pipeline or `ColumnTransformer`, instead of manually transforming columns outside the pipeline.
- For **right-skewed** data (long tail on the right, like income), a log transform (`np.log1p`) compresses large values and pulls the distribution toward normal. `log1p` is used instead of plain `log` so it can safely handle zero values.
- For **left-skewed** data (long tail on the left, like a bounded score), squaring the values (`np.square`) has the opposite compressing effect and pushes it back toward normal.
- These custom transforms still need scaling afterward, since transforming shape doesn't fix the scale — so `FunctionTransformer` is usually paired with `StandardScaler` in its own mini `Pipeline`.
- For **heavily skewed** data where a manual log/square isn't enough, `PowerTransformer` automatically finds the best transformation to normalize the distribution — and it applies standardization internally, so a separate `StandardScaler` isn't needed after it.
- `PowerTransformer` has two methods: `box-cox`, which only works on strictly positive data (no zeros or negatives), and `yeo-johnson`, which works on any data including zeros and negatives — making it the more flexible, general-purpose choice.
- Nesting pipelines (a `Pipeline` per feature group, all combined inside one `ColumnTransformer`) keeps preprocessing organized when different columns each need a different multi-step treatment (encode → transform → scale).

**Code:** [function_transformer.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-09-Function_Transformer/function_transformer.ipynb)

---

### Day 10: Pipelines
**What I learned:**
- A `Pipeline` chains multiple preprocessing steps and a final model into one object, so instead of manually calling `fit_transform` at every stage, the whole flow (impute → encode → scale → model) runs with a single `.fit()` and `.predict()` call.
- Feature engineering before pipelining still matters — e.g. combining `sibsp` and `parch` into a single `family` column reduces redundant features and can capture the relationship better than two separate columns.
- Chaining multiple `ColumnTransformer` steps works because each one passes its full output (transformed + passthrough columns) to the next step — but this means column *positions* shift after each step, so you have to track indexes carefully (which is why the notebook prints `.get_feature_names_out()` after each stage to verify correct indexing).
- `remainder='passthrough'` at every stage is what keeps untouched columns flowing through the pipeline instead of getting dropped between steps.
- A fitted pipeline's internal steps stay inspectable — e.g. `pipe.named_steps['ohe']` lets you go back and check exactly which columns a specific step acted on, which is useful for debugging a multi-stage pipeline.
- For production use, the entire fitted pipeline (preprocessing + model together) can be serialized with `pickle.dump()` into a `.pkl` file — this saves not just the trained model but also every fitted transformer (imputer statistics, encoder categories, scaler ranges), so new raw data can be fed directly into `production.predict()` without redoing any preprocessing manually.
- This is the real value of pipelines: the exact preprocessing logic used during training is guaranteed to be reapplied identically at prediction time, since it's bundled into one saved object.

**Code:** [pipeline.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-10-PipeLines/pipeline.ipynb)

---

### Day 11: Binning & Binarization
**What I learned:**
- Binning converts a continuous numeric column into discrete categories/buckets — useful when the raw numeric value matters less than the *range* it falls into (e.g. age groups instead of exact age).
- `KBinsDiscretizer` automates this: `n_bins` sets how many buckets to create, `encode='ordinal'` outputs the bin number as an integer, and `strategy='quantile'` ensures each bin gets roughly the same number of data points (as opposed to equal-width bins, which can leave some bins nearly empty if the data is skewed).
- After fitting, `.bin_edges_` shows the actual boundary values used to split the data into bins — useful for understanding what each bin number actually represents.
- Binning was tested by comparing model performance (via `cross_val_score`) on the raw features vs. the binned features, to see if grouping the values into ranges helps a `DecisionTreeClassifier` find patterns more easily than raw continuous values.
- Binarization is a simpler, specific case of binning — it converts a numeric column into just two categories (0 or 1) based on a single threshold.
- `Binarizer(threshold=X)` outputs 1 for values above the threshold and 0 for values at or below it — used here to create simple flags like `Adult` (age ≥ 18) and `senior_citizen` (age ≥ 60) directly from a continuous age column.
- Both binning and binarization are fit on training data and applied to test data using `.transform()`, keeping the same bin boundaries/threshold consistent across both sets.

**Code:** [binning_binarization.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-11-Binning-Binarization/binning_binarization.ipynb)

---

### Day 12: Linear Regression (OLS)
**What I learned:**
- Built Linear Regression from scratch using the **Ordinary Least Squares (OLS)** closed-form solution instead of relying on `sklearn`, to understand what's happening under the hood.
- The OLS formula finds the coefficients (betas) that minimize the sum of squared errors directly, using the matrix equation: `β = (XᵀX)⁻¹ Xᵀy` — no iteration or learning rate needed, unlike gradient descent.
- Before applying the formula, a column of 1s is inserted at the start of `X` (`np.insert(..., 0, 1, axis=1)`) — this accounts for the intercept term, since the intercept is really just the coefficient of a constant feature equal to 1.
- After solving for `betas`, the first value is the **intercept** and the rest are the **feature coefficients** — separating them out (`betas[0]` vs `betas[1:]`) makes the model usable for prediction the same way `sklearn`'s model is.
- Prediction is just the linear equation applied to new data: `y = Xβ + intercept`, implemented directly with a dot product.
- Validated the custom implementation by comparing its coefficients and Mean Absolute Error against `sklearn`'s built-in `LinearRegression` on the same data — matching results confirm the from-scratch math is correct.
- OLS works well here because it's a single feature (simple linear regression) with a small, well-behaved dataset — for larger feature sets, computing `(XᵀX)⁻¹` directly becomes computationally expensive, which is where iterative methods like gradient descent become more practical.

**Code:** [MyLR.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-12-Linear-Regressor-using-OLS/MyLR.ipynb)

---

### Day 13: Gradient Descent
**What I learned:**
- Unlike OLS (which solves for coefficients directly via a matrix formula), Gradient Descent finds them **iteratively** — starting with initial guesses and updating them step by step to reduce error. This scales much better to large datasets/many features where computing `(XᵀX)⁻¹` directly gets expensive.
- The core update rule is the same across all variants: `param = param - learning_rate * gradient`, where the gradient tells you the direction of steepest increase in error, so moving opposite to it reduces the loss.
- **Batch Gradient Descent (BGD)** computes the gradient using the *entire* training set at every epoch. This gives a stable, accurate update direction each step, but it's slow per epoch since it processes all data before updating even once.
- **Stochastic Gradient Descent (SGD)** updates the coefficients using just *one random row* at a time, for every row in the dataset per epoch. This makes each update noisier (less accurate direction) but much faster and able to escape shallow local minima due to that noise.
- **Mini-Batch Gradient Descent (MBGD)** is the middle ground — it updates coefficients using a small random *batch* of rows at a time, balancing the stability of BGD with the speed of SGD.
- The `learning_rate` and `epochs` are critical hyperparameters: too high a learning rate can overshoot the minimum, too low takes too many epochs to converge — this shows up directly in this notebook, where BGD needed ~1000 epochs at `lr=0.001` to get a good R² score, while a well-tuned SGD converged in far fewer epochs.
- Validated all three custom implementations by comparing their R² scores against each other and against `sklearn`'s built-in `LinearRegression`/`SGDRegressor` on the same synthetic dataset.
- `sklearn`'s `SGDRegressor` has a `.partial_fit()` method, which lets you manually feed it batches — used here to replicate true mini-batch behavior even with `sklearn`'s SGD implementation.

**Code:** [Gradient_Descent.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-13-Gradient-Descent/Gradient_Descent.ipynb)

---

### Day 14: Polynomial Regression
**What I learned:**
- Plain Linear Regression fits a straight line, but real relationships (like a projectile's trajectory) are often curved — Polynomial Regression handles this by adding powers of the input feature (`t`, `t²`, `t³`...) as new features, then fitting a linear model on top of those expanded features.
- `PolynomialFeatures(degree=n)` transforms a single feature into `n` polynomial terms — the model itself is still `LinearRegression`, it's the *input features* that become non-linear, not the model.
- Used a simulated physics dataset (projectile motion, with a known true formula `d = v₀t - 0.5gt²`) with added noise — this makes it possible to visually and numerically compare how close each fitted curve gets to the actual underlying pattern.
- Comparing degrees `1, 2, 5, 10` side by side showed the **underfitting vs overfitting** tradeoff directly: degree 1 (straight line) can't capture the curve at all (underfitting), degree 2 matches the true physics almost exactly, and higher degrees like 10 start fitting noise in the training data rather than the actual trend (overfitting).
- The real physics here is degree-2 (`t²` term from gravity), which is why degree 2 gave the best fit — a good reminder that the right polynomial degree should reflect the actual complexity of the relationship, not just be pushed higher for a "better fit."
- Comparing **train MSE vs test MSE** for each degree makes overfitting visible numerically, not just visually — overfit models show low train error but noticeably higher test error, since they've started memorizing noise instead of learning the general pattern.

**Code:** [polynomial_regressor.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-14-Polynomial-Regression/polynomial_regressor.ipynb)

---

### Day 15: Condition Number & VIF
**What I learned:**
- **Multicollinearity** is when input features are highly correlated with each other — this doesn't hurt a model's overall accuracy much, but it makes individual coefficients unstable and hard to interpret, since the model can't tell which correlated feature is "responsible" for the effect.
- **VIF (Variance Inflation Factor)** quantifies multicollinearity per feature — a high VIF means that feature can be well-predicted from the other features, meaning it's carrying redundant information.
- **Condition Number** (`np.linalg.cond`) is a single number summarizing multicollinearity across the *entire* feature matrix — a very high condition number signals the matrix is close to singular, which makes the OLS solution numerically unstable.
- Used the California Housing dataset to find two clearly correlated pairs: `AveRooms`/`AveBedrms` and `Latitude`/`Longitude` — confirmed via `.corr()` and high VIF scores.
- For truly redundant columns (`AveRooms`/`AveBedrms`), engineered a single `BedroomRatio` feature instead of just dropping one column — this keeps the information from both columns instead of throwing half of it away.
- For `Latitude`/`Longitude`, dropping either wasn't an option since location is clearly relevant to house price — instead, used domain knowledge to engineer `dist_to_LA` and `dist_to_SF` (Euclidean distance to major cities), which captures the same locational signal without the raw coordinate correlation.
- **Feature scaling also affects the condition number** — standardizing the features with `StandardScaler` dropped the condition number significantly, even independent of the multicollinearity fixes, since unscaled features with very different ranges also inflate the condition number.
- Comparing MSE before and after fixing multicollinearity + scaling confirmed the fix: prediction error dropped, showing that resolving multicollinearity doesn't just help interpretability, it can improve actual model performance too.

**Code:** [condition_number_VIF.ipynb](https://github.com/KevalSolanki77/My-ML-journey/blob/main/Day-15-Condition-Numbers/condition_number_VIF.ipynb)


---

## 🛠️ Tech Stack
`Python` `NumPy` `Pandas` `Matplotlib` `Seaborn` `Scikit-learn`

## License 
[MIT License](https://github.com/KevalSolanki77/My-ML-journey/blob/main/LICENSE)
