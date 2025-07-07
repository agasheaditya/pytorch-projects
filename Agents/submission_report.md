### Code Block 1
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Output:
```
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
```

### Code Block 2
```python
data.head(5)
```

#### Output:
```
DataFrame Output:
   Unnamed: 0 case_id continent education_of_employee has_job_experience requires_job_training  no_of_employees  yr_of_estab region_of_employment  prevailing_wage unit_of_wage full_time_position case_status
0           0  EZYV01      Asia           High School                  N                     N            14513         2007                 West         592.2029         Hour                  Y      Denied
1           1  EZYV02      Asia              Master's                  Y                     N             2412         2002            Northeast       83425.6500         Year                  Y   Certified
2           2  EZYV03      Asia            Bachelor's                  N                     Y            44444         2008                 West      122996.8600         Year                  Y      Denied
3           3  EZYV04      Asia            Bachelor's                  N                     N               98         1897                 West       83434.0300         Year                  Y      Denied
4           4  EZYV05    Africa              Master's                  Y                     N             1082         2005                South      149907.3900         Year                  Y   Certified
```

### Code Block 3
```python
data.tail(5)
```

#### Output:
```
DataFrame Output:
   Unnamed: 0    case_id continent education_of_employee has_job_experience requires_job_training  no_of_employees  yr_of_estab region_of_employment  prevailing_wage unit_of_wage full_time_position case_status
0       25475  EZYV25476      Asia            Bachelor's                  Y                     Y             2601         2008                South         77092.57         Year                  Y   Certified
1       25476  EZYV25477      Asia           High School                  Y                     N             3274         2006            Northeast        279174.79         Year                  Y   Certified
2       25477  EZYV25478      Asia              Master's                  Y                     N             1121         1910                South        146298.85         Year                  N   Certified
3       25478  EZYV25479      Asia              Master's                  Y                     Y             1918         1887                 West         86154.77         Year                  Y   Certified
4       25479  EZYV25480      Asia            Bachelor's                  Y                     N             3195         1960              Midwest         70876.91         Year                  Y   Certified
```

### Code Block 4
```python
data.shape
```

#### Output:
```
(25480, 12)
```

### Code Block 5
```python
data.info()
```

#### Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25480 entries, 0 to 25479
Data columns (total 12 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   case_id                25480 non-null  object 
 1   continent              25480 non-null  object 
 2   education_of_employee  25480 non-null  object 
 3   has_job_experience     25480 non-null  object 
 4   requires_job_training  25480 non-null  object 
 5   no_of_employees        25480 non-null  int64  
 6   yr_of_estab            25480 non-null  int64  
 7   region_of_employment   25480 non-null  object 
 8   prevailing_wage        25480 non-null  float64
 9   unit_of_wage           25480 non-null  object 
 10  full_time_position     25480 non-null  object 
 11  case_status            25480 non-null  object 
dtypes: float64(1), int64(2), object(9)
memory usage: 2.3+ MB
```

### Code Block 6
```python
data.describe(include="all").T
```

#### Output:
```
DataFrame Output:
              Unnamed: 0    count   unique         top     freq  mean  std  min  25%  50%  75%  max
0                case_id  25480.0  25480.0   EZYV25480      1.0   NaN  NaN  NaN  NaN  NaN  NaN  NaN
1              continent  25480.0      6.0        Asia  16861.0   NaN  NaN  NaN  NaN  NaN  NaN  NaN
2  education_of_employee  25480.0      4.0  Bachelor's  10234.0   NaN  NaN  NaN  NaN  NaN  NaN  NaN
3     has_job_experience  25480.0      2.0           Y  14802.0   NaN  NaN  NaN  NaN  NaN  NaN  NaN
4  requires_job_training  25480.0      2.0           N  22525.0   NaN  NaN  NaN  NaN  NaN  NaN  NaN
```

### Code Block 7
```python
negative_count = (data['no_of_employees'] < 0).sum()
print(f"Number of negative entries: {negative_count}")
```

#### Output:
```
Number of negative entries: 33
```

### Code Block 8
```python
negative_count = (data['no_of_employees'] < 0).sum()
print(f"Number of negative entries: {negative_count}")
```

#### Output:
```
Number of negative entries: 0
```

### Code Block 9
```python
data.info()
```

#### Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25480 entries, 0 to 25479
Data columns (total 12 columns):
 #   Column                 Non-Null Count  Dtype   
---  ------                 --------------  -----   
 0   case_id                25480 non-null  object  
 1   continent              25480 non-null  category
 2   education_of_employee  25480 non-null  category
 3   has_job_experience     25480 non-null  category
 4   requires_job_training  25480 non-null  category
 5   no_of_employees        25480 non-null  float64 
 6   yr_of_estab            25480 non-null  int64   
 7   region_of_employment   25480 non-null  category
 8   prevailing_wage        25480 non-null  float64 
 9   unit_of_wage           25480 non-null  category
 10  full_time_position     25480 non-null  category
 11  case_status            25480 non-null  category
dtypes: category(8), float64(2), int64(1), object(1)
memory usage: 996.7+ KB
```

### Code Block 10
```python
categorical_cols = data.select_dtypes(include=['category']).columns

for col in categorical_cols:
    print(f"\nValue counts for column: {col}")
    print(data[col].value_counts())
```

#### Output:
```
Value counts for column: continent
continent
Asia             16861
Europe            3732
North America     3292
South America      852
Africa             551
Oceania            192
Name: count, dtype: int64

Value counts for column: education_of_employee
education_of_employee
Bachelor's     10234
Master's        9634
High School     3420
Doctorate       2192
Name: count, dtype: int64

Value counts for column: has_job_experience
has_job_experience
Y    14802
N    10678
Name: count, dtype: int64

Value counts for column: requires_job_training
requires_job_training
N    22525
Y     2955
Name: count, dtype: int64

Value counts for column: region_of_employment
region_of_employment
Northeast    7195
South        7017
West         6586
Midwest      4307
Island        375
Name: count, dtype: int64

Value counts for column: unit_of_wage
unit_of_wage
Year     22962
Hour      2157
Week       272
Month       89
Name: count, dtype: int64

Value counts for column: full_time_position
full_time_position
Y    22773
N     2707
Name: count, dtype: int64

Value counts for column: case_status
case_status
Certified    17018
Denied        8462
Name: count, dtype: int64
```

### Code Block 11
```python
#data['education_of_employee'] = data['education_of_employee'].astype('category')
le = LabelEncoder()

#converting below to numbers so that above function histogram_boxplot can be used to call mean without error
data['education_of_employee'] = le.fit_transform(data['education_of_employee'])

histogram_boxplot(data, "education_of_employee", figsize=(15, 10))
```

#### Output Image:
![Generated Plot](extracted_images\image_1.png)

### Code Block 12
```python
labeled_barplot(data, 'education_of_employee')
```

#### Output:
```
<ipython-input-202-1e58b03272cc>:22: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  ax = sns.countplot(
```

### Code Block 13
```python
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)
```

#### Output:
```
{"Bachelor's": np.int64(0), 'Doctorate': np.int64(1), 'High School': np.int64(2), "Master's": np.int64(3)}
```

### Code Block 14
```python
#converting below to numbers so that above function histogram_boxplot can be used to call mean without error
data['region_of_employment'] = le.fit_transform(data['region_of_employment'])

histogram_boxplot(data, "region_of_employment", figsize=(15, 10))
```

#### Output Image:
![Generated Plot](extracted_images\image_2.png)

### Code Block 15
```python
labeled_barplot(data, 'region_of_employment')
```

#### Output:
```
<ipython-input-202-1e58b03272cc>:22: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  ax = sns.countplot(
```

### Code Block 16
```python
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)
```

#### Output:
```
{'Island': np.int64(0), 'Midwest': np.int64(1), 'Northeast': np.int64(2), 'South': np.int64(3), 'West': np.int64(4)}
```

### Code Block 17
```python
#converting below to numbers so that above function histogram_boxplot can be used to call mean without error
data['has_job_experience'] = le.fit_transform(data['has_job_experience'])

histogram_boxplot(data, "has_job_experience", figsize=(15, 10))
```

#### Output Image:
![Generated Plot](extracted_images\image_3.png)

### Code Block 18
```python
labeled_barplot(data, 'has_job_experience')
```

#### Output:
```
<ipython-input-202-1e58b03272cc>:22: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  ax = sns.countplot(
```

### Code Block 19
```python
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)
```

#### Output:
```
{'N': np.int64(0), 'Y': np.int64(1)}
```

### Code Block 20
```python
data['case_status'] = le.fit_transform(data['case_status'])

histogram_boxplot(data, "case_status", figsize=(15, 10))
```

#### Output Image:
![Generated Plot](extracted_images\image_4.png)

### Code Block 21
```python
labeled_barplot(data, 'case_status')
```

#### Output:
```
<ipython-input-202-1e58b03272cc>:22: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  ax = sns.countplot(
```

### Code Block 22
```python
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(mapping)
```

#### Output:
```
{'Certified': np.int64(0), 'Denied': np.int64(1)}
```

### Code Block 23
```python
distribution_plot_wrt_target(data, "education_of_employee", "case_status")
```

#### Output:
```
<ipython-input-215-d405489ef7b9>:31: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
<ipython-input-215-d405489ef7b9>:34: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
```

### Code Block 24
```python
stacked_barplot(data, "education_of_employee", "case_status")
```

#### Output:
```
case_status                0     1    All
education_of_employee                    
All                    17018  8462  25480
0                       6367  3867  10234
2                       1164  2256   3420
3                       7575  2059   9634
1                       1912   280   2192
------------------------------------------------------------------------------------------------------------------------
```

### Code Block 25
```python
distribution_plot_wrt_target(data, "continent", "case_status")
```

#### Output:
```
<ipython-input-215-d405489ef7b9>:31: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
<ipython-input-215-d405489ef7b9>:34: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
```

### Code Block 26
```python
stacked_barplot(data, "continent", "case_status")
```

#### Output:
```
case_status        0     1    All
continent                        
All            17018  8462  25480
Asia           11012  5849  16861
North America   2037  1255   3292
Europe          2957   775   3732
South America    493   359    852
Africa           397   154    551
Oceania          122    70    192
------------------------------------------------------------------------------------------------------------------------
```

### Code Block 27
```python
distribution_plot_wrt_target(data, "has_job_experience", "case_status")
```

#### Output:
```
<ipython-input-215-d405489ef7b9>:31: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
<ipython-input-215-d405489ef7b9>:34: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
```

### Code Block 28
```python
stacked_barplot(data, "has_job_experience", "case_status")
```

#### Output:
```
case_status             0     1    All
has_job_experience                    
All                 17018  8462  25480
0                    5994  4684  10678
1                   11024  3778  14802
------------------------------------------------------------------------------------------------------------------------
```

### Code Block 29
```python
distribution_plot_wrt_target(data, "prevailing_wage", "region_of_employment")
```

#### Output:
```
<ipython-input-215-d405489ef7b9>:31: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
<ipython-input-215-d405489ef7b9>:34: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
```

### Code Block 30
```python
distribution_plot_wrt_target(data, "prevailing_wage", "case_status")
```

#### Output:
```
<ipython-input-215-d405489ef7b9>:31: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
<ipython-input-215-d405489ef7b9>:34: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
```

### Code Block 31
```python
distribution_plot_wrt_target(data, "unit_of_wage", "case_status")
```

#### Output:
```
<ipython-input-215-d405489ef7b9>:31: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")
<ipython-input-215-d405489ef7b9>:34: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(
```

### Code Block 32
```python
stacked_barplot(data, "unit_of_wage", "case_status")
```

#### Output:
```
case_status       0     1    All
unit_of_wage                    
All           17018  8462  25480
Year          16047  6915  22962
Hour            747  1410   2157
Week            169   103    272
Month            55    34     89
------------------------------------------------------------------------------------------------------------------------
```

### Code Block 33
```python
# checking missing values in the data
data.isna().sum()
```

#### Output:
```
DataFrame Output:
              Unnamed: 0  0
0                case_id  0
1              continent  0
2  education_of_employee  0
3     has_job_experience  0
4  requires_job_training  0
```

### Code Block 34
```python
# Case status vs Education of employee
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='case_status', y='education_of_employee');
```

#### Output Image:
![Generated Plot](extracted_images\image_5.png)

### Code Block 35
```python
# Case status vs has_job_experience
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='case_status', y='has_job_experience');
```

#### Output Image:
![Generated Plot](extracted_images\image_6.png)

### Code Block 36
```python
# "unit_of_wage", "case_status"
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='unit_of_wage', y='case_status');
```

#### Output Image:
![Generated Plot](extracted_images\image_7.png)

### Code Block 37
```python
# checking the distribution of the target variable
data["case_status"].value_counts(1)
```

#### Output:
```
DataFrame Output:
  Unnamed: 0_level_0         proportion
         case_status Unnamed: 1_level_1
0                  0           0.667896
1                  1           0.332104
```

### Code Block 38
```python
# creating dummy variables
X = pd.get_dummies(X, columns=X.select_dtypes(include=["object", "category"]).columns.tolist(), drop_first=True)

# specifying the datatype of the independent variables data frame
X = X.astype(float)

X.head()
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  education_of_employee  has_job_experience  no_of_employees  yr_of_estab  region_of_employment  prevailing_wage  continent_Asia  continent_Europe  continent_North America  continent_Oceania  continent_South America  requires_job_training_Y  unit_of_wage_Month  unit_of_wage_Week  unit_of_wage_Year  full_time_position_Y
0           0                    2.0                 0.0          14513.0       2007.0                   4.0         592.2029             1.0               0.0                      0.0                0.0                      0.0                      0.0                 0.0                0.0                0.0                   1.0
1           1                    3.0                 1.0           2412.0       2002.0                   2.0       83425.6500             1.0               0.0                      0.0                0.0                      0.0                      0.0                 0.0                0.0                1.0                   1.0
2           2                    0.0                 0.0          44444.0       2008.0                   4.0      122996.8600             1.0               0.0                      0.0                0.0                      0.0                      1.0                 0.0                0.0                1.0                   1.0
3           3                    0.0                 0.0             98.0       1897.0                   4.0       83434.0300             1.0               0.0                      0.0                0.0                      0.0                      0.0                 0.0                0.0                1.0                   1.0
4           4                    3.0                 1.0           1082.0       2005.0                   3.0      149907.3900             0.0               0.0                      0.0                0.0                      0.0                      0.0                 0.0                0.0                1.0                   1.0
```

### Code Block 39
```python
#no need to change y as it is already 0 or 1
y.head()
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  case_status
0           0            1
1           1            0
2           2            1
3           3            1
4           4            0
```

### Code Block 40
```python
print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape, '\n')
print("Percentage of classes in training set:")
print(100*y_train.value_counts(normalize=True), '\n')
print("Percentage of classes in test set:")
print(100*y_test.value_counts(normalize=True))
```

#### Output:
```
Shape of training set: (20384, 16)
Shape of test set: (5096, 16) 

Percentage of classes in training set:
case_status
0    66.787677
1    33.212323
Name: proportion, dtype: float64 

Percentage of classes in test set:
case_status
0    66.797488
1    33.202512
Name: proportion, dtype: float64
```

### Code Block 41
```python
# creating an instance of the decision tree model
dtree1 = DecisionTreeClassifier(random_state=42)    # random_state sets a seed value and enables reproducibility

# fitting the model to the training data
dtree1.fit(X_train, y_train)
```

#### Output:
```
DecisionTreeClassifier(random_state=42)
```

### Code Block 42
```python
confusion_matrix_sklearn(dtree1, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_8.png)

### Code Block 43
```python
dtree1_train_perf = model_performance_classification_sklearn(
    dtree1, X_train, y_train
)
dtree1_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy  Recall  Precision   F1
0           0       1.0     1.0        1.0  1.0
```

### Code Block 44
```python
confusion_matrix_sklearn(dtree1, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_9.png)

### Code Block 45
```python
dtree1_test_perf = model_performance_classification_sklearn(
    dtree1, X_test, y_test
)
dtree1_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.648744  0.470449   0.471006  0.470727
```

### Code Block 46
```python
# list of feature names in X_train
feature_names = list(X_train.columns)

# set the figure size for the plot
plt.figure(figsize=(20, 20))

# plotting the decision tree
out = tree.plot_tree(
    dtree1,                         # decision tree classifier model
    feature_names=feature_names,    # list of feature names (columns) in the dataset
    filled=True,                    # fill the nodes with colors based on class
    fontsize=9,                     # font size for the node text
    node_ids=False,                 # do not show the ID of each node
    class_names=None,               # whether or not to display class names
)

# add arrows to the decision tree splits if they are missing
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor("black")    # set arrow color to black
        arrow.set_linewidth(1)          # set arrow linewidth to 1

# displaying the plot
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_10.png)

### Code Block 47
```python
# creating an instance of the best model
dtree2 = best_estimator

# fitting the best model to the training data
dtree2.fit(X_train, y_train)
```

#### Output:
```
DecisionTreeClassifier(max_depth=np.int64(8), max_leaf_nodes=np.int64(10),
                       min_samples_split=np.int64(10), random_state=42)
```

### Code Block 48
```python
confusion_matrix_sklearn(dtree2, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_11.png)

### Code Block 49
```python
dtree2_train_perf = model_performance_classification_sklearn(
    dtree2, X_train, y_train
)
dtree2_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.732094  0.566765   0.602828  0.584241
```

### Code Block 50
```python
confusion_matrix_sklearn(dtree2, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_12.png)

### Code Block 51
```python
dtree2_test_perf = model_performance_classification_sklearn(
    dtree2, X_test, y_test
)
dtree2_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.731358  0.560875    0.60254  0.580961
```

### Code Block 52
```python
#base_estimator for bagging classifier is a decision tree by default
bagging_estimator=BaggingClassifier(random_state=1)
bagging_estimator.fit(X_train,y_train)
```

#### Output:
```
BaggingClassifier(random_state=1)
```

### Code Block 53
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(bagging_estimator, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_13.png)

### Code Block 54
```python
bagging_estimator_train_perf = model_performance_classification_sklearn(
    bagging_estimator, X_train, y_train
)
bagging_estimator_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.980033  0.945495   0.994098  0.969188
```

### Code Block 55
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(bagging_estimator, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_14.png)

### Code Block 56
```python
bagging_estimator_test_perf = model_performance_classification_sklearn(
    bagging_estimator, X_test, y_test
)
bagging_estimator_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.703297  0.422577      0.572  0.486064
```

### Code Block 57
```python
#Train the random forest classifier
rf_estimator=RandomForestClassifier(random_state=1)
rf_estimator.fit(X_train,y_train)
```

#### Output:
```
RandomForestClassifier(random_state=1)
```

### Code Block 58
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(rf_estimator, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_15.png)

### Code Block 59
```python
rf_estimator_train_perf = model_performance_classification_sklearn(
    rf_estimator, X_train, y_train
)
rf_estimator_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy  Recall  Precision   F1
0           0       1.0     1.0        1.0  1.0
```

### Code Block 60
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(rf_estimator, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_16.png)

### Code Block 61
```python
rf_estimator_test_perf = model_performance_classification_sklearn(
    rf_estimator, X_test, y_test
)
rf_estimator_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.720958  0.469858   0.602273  0.527888
```

### Code Block 62
```python
adaboost = AdaBoostClassifier(random_state=1)
adaboost.fit(X_train,y_train)
```

#### Output:
```
AdaBoostClassifier(random_state=1)
```

### Code Block 63
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(adaboost, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_17.png)

### Code Block 64
```python
adaboost_train_perf = model_performance_classification_sklearn(
    adaboost, X_train, y_train
)
adaboost_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision       F1
0           0  0.733713  0.387888   0.671611  0.49176
```

### Code Block 65
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(adaboost, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_18.png)

### Code Block 66
```python
adaboost_test_perf = model_performance_classification_sklearn(
    adaboost, X_test, y_test
)
adaboost_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.728218  0.381206   0.656155  0.482243
```

### Code Block 67
```python
gboost = GradientBoostingClassifier(random_state=1)
gboost.fit(X_train,y_train)
```

#### Output:
```
GradientBoostingClassifier(random_state=1)
```

### Code Block 68
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(gboost, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_19.png)

### Code Block 69
```python
gboost_train_perf = model_performance_classification_sklearn(
    gboost, X_train, y_train
)
gboost_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.754857  0.512555    0.67157  0.581386
```

### Code Block 70
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(gboost, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_20.png)

### Code Block 71
```python
gboost_test_perf = model_performance_classification_sklearn(
    gboost, X_test, y_test
)
gboost_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision       F1
0           0  0.742739  0.497636   0.646201  0.56227
```

### Code Block 72
```python
xgboost = XGBClassifier(random_state=1,eval_metric='logloss')
xgboost.fit(X_train, y_train)
```

#### Output:
```
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=1, ...)
```

### Code Block 73
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(xgboost, X_train, y_train)
```

#### Output Image:
![Generated Plot](extracted_images\image_21.png)

### Code Block 74
```python
xgboost_train_perf = model_performance_classification_sklearn(
    xgboost, X_train, y_train
)
xgboost_train_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.839973  0.660709   0.822545  0.732798
```

### Code Block 75
```python
#Using above defined function to get accuracy, recall and precision on train and test set
confusion_matrix_sklearn(xgboost, X_test, y_test)
```

#### Output Image:
![Generated Plot](extracted_images\image_22.png)

### Code Block 76
```python
xgboost_test_perf = model_performance_classification_sklearn(
    xgboost, X_test, y_test
)
xgboost_test_perf
```

#### Output:
```
DataFrame Output:
   Unnamed: 0  Accuracy    Recall  Precision        F1
0           0  0.724294  0.462175   0.612373  0.526777
```

### Code Block 77
```python
# Splitting data into training, validation and test set:

# first we split data into 2 parts, say temporary and test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y
)

# then we split the temporary set into train and validation
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.4, random_state=0, stratify=y_temp
)

print(X_train.shape, X_val.shape, X_test.shape)
```

#### Output:
```
(12740, 16) (7644, 16) (5096, 16)
```

### Code Block 78
```python
# Checking class balance for whole data, train set, validation set, and test set

print("Target value ratio in y")
print(y.value_counts(1))
print("*" * 80)
print("Target value ratio in y_train")
print(y_train.value_counts(1))
print("*" * 80)
print("Target value ratio in y_val")
print(y_val.value_counts(1))
print("*" * 80)
print("Target value ratio in y_test")
print(y_test.value_counts(1))
print("*" * 80)
```

#### Output:
```
Target value ratio in y
case_status
0    0.667896
1    0.332104
Name: proportion, dtype: float64
********************************************************************************
Target value ratio in y_train
case_status
0    0.667896
1    0.332104
Name: proportion, dtype: float64
********************************************************************************
Target value ratio in y_val
case_status
0    0.667844
1    0.332156
Name: proportion, dtype: float64
********************************************************************************
Target value ratio in y_test
case_status
0    0.667975
1    0.332025
Name: proportion, dtype: float64
********************************************************************************
```

### Code Block 79
```python
print("Before OverSampling, count of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, count of label '0': {} \n".format(sum(y_train == 0)))

print("After OverSampling, count of label '1': {}".format(sum(y_train_over == 1)))
print("After OverSampling, count of label '0': {} \n".format(sum(y_train_over == 0)))

print("After OverSampling, the shape of train_X: {}".format(X_train_over.shape))
print("After OverSampling, the shape of train_y: {} \n".format(y_train_over.shape))
```

#### Output:
```
Before OverSampling, count of label '1': 4231
Before OverSampling, count of label '0': 8509 

After OverSampling, count of label '1': 8509
After OverSampling, count of label '0': 8509 

After OverSampling, the shape of train_X: (17018, 16)
After OverSampling, the shape of train_y: (17018,)
```

### Code Block 80
```python
# training the decision tree model with oversampled training set
dtree2.fit(X_train_over, y_train_over)
```

#### Output:
```
DecisionTreeClassifier(max_depth=np.int64(8), max_leaf_nodes=np.int64(10),
                       min_samples_split=np.int64(10), random_state=42)
```

### Code Block 81
```python
# Checking recall score on oversampled train and validation set
print(recall_score(y_train_over, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.7313432835820896
0.551792044111855
```

### Code Block 82
```python
# Confusion matrix for oversampled train data
cm = confusion_matrix(y_train_over, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 83
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_23.png)

### Code Block 84
```python
# training the Bagging estimator with oversampled training set
bagging_estimator.fit(X_train_over, y_train_over)
```

#### Output:
```
BaggingClassifier(random_state=1)
```

### Code Block 85
```python
# Checking recall score on oversampled train and validation set
print(recall_score(y_train_over, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.9734398871782818
0.435998424576605
```

### Code Block 86
```python
# Confusion matrix for oversampled train data
cm = confusion_matrix(y_train_over, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 87
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_24.png)

### Code Block 88
```python
# training the Bagging estimator with oversampled training set
rf_estimator.fit(X_train_over, y_train_over)
```

#### Output:
```
RandomForestClassifier(random_state=1)
```

### Code Block 89
```python
# Checking recall score on oversampled train and validation set
print(recall_score(y_train_over, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.9998824773768951
0.49192595510043324
```

### Code Block 90
```python
# Confusion matrix for oversampled train data
cm = confusion_matrix(y_train_over, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 91
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_25.png)

### Code Block 92
```python
# training the Bagging estimator with oversampled training set
adaboost.fit(X_train_over, y_train_over)
```

#### Output:
```
AdaBoostClassifier(random_state=1)
```

### Code Block 93
```python
# Checking recall score on oversampled train and validation set
print(recall_score(y_train_over, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.7453284757315783
0.5265852697912564
```

### Code Block 94
```python
# Confusion matrix for oversampled train data
cm = confusion_matrix(y_train_over, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 95
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_26.png)

### Code Block 96
```python
# training the Bagging estimator with oversampled training set
gboost.fit(X_train_over, y_train_over)
```

#### Output:
```
GradientBoostingClassifier(random_state=1)
```

### Code Block 97
```python
# Checking recall score on oversampled train and validation set
print(recall_score(y_train_over, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.7630743918204255
0.5246159905474597
```

### Code Block 98
```python
# Confusion matrix for oversampled train data
cm = confusion_matrix(y_train_over, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 99
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_27.png)

### Code Block 100
```python
# training the Bagging estimator with oversampled training set
xgboost.fit(X_train_over, y_train_over)
```

#### Output:
```
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=1, ...)
```

### Code Block 101
```python
# Checking recall score on oversampled train and validation set
print(recall_score(y_train_over, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.8419320719238453
0.4970460811343048
```

### Code Block 102
```python
# Confusion matrix for oversampled train data
cm = confusion_matrix(y_train_over, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 103
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_28.png)

### Code Block 104
```python
print("Before Under Sampling, count of label '1': {}".format(sum(y_train == 1)))
print("Before Under Sampling, count of label '0': {} \n".format(sum(y_train == 0)))

print("After Under Sampling, count of label '1': {}".format(sum(y_train_un == 1)))
print("After Under Sampling, count of label '0': {} \n".format(sum(y_train_un == 0)))

print("After Under Sampling, the shape of train_X: {}".format(X_train_un.shape))
print("After Under Sampling, the shape of train_y: {} \n".format(y_train_un.shape))
```

#### Output:
```
Before Under Sampling, count of label '1': 4231
Before Under Sampling, count of label '0': 8509 

After Under Sampling, count of label '1': 4231
After Under Sampling, count of label '0': 4231 

After Under Sampling, the shape of train_X: (8462, 16)
After Under Sampling, the shape of train_y: (8462,)
```

### Code Block 105
```python
# training the decision tree model with undersampled training set
dtree2.fit(X_train_un, y_train_un)
```

#### Output:
```
DecisionTreeClassifier(max_depth=np.int64(8), max_leaf_nodes=np.int64(10),
                       min_samples_split=np.int64(10), random_state=42)
```

### Code Block 106
```python
# Checking recall score on undersampled train and validation set
print(recall_score(y_train_un, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.6908532261876625
0.6738873572272548
```

### Code Block 107
```python
# Confusion matrix for undersampled train data
cm = confusion_matrix(y_train_un, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 108
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_29.png)

### Code Block 109
```python
# training the Bagging estimator with undrsampled training set
bagging_estimator.fit(X_train_un, y_train_un)
```

#### Output:
```
BaggingClassifier(random_state=1)
```

### Code Block 110
```python
# Checking recall score on undersample train and validation set
print(recall_score(y_train_un, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.9718742614039234
0.6309570697124852
```

### Code Block 111
```python
# Confusion matrix for undersampled train data
cm = confusion_matrix(y_train_un, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 112
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_30.png)

### Code Block 113
```python
# training the Bagging estimator with undersampled training set
rf_estimator.fit(X_train_un, y_train_un)
```

#### Output:
```
RandomForestClassifier(random_state=1)
```

### Code Block 114
```python
# Checking recall score on undersampled train and validation set
print(recall_score(y_train_un, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
1.0
0.6849153209925167
```

### Code Block 115
```python
# Confusion matrix for undersampled train data
cm = confusion_matrix(y_train_un, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 116
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_31.png)

### Code Block 117
```python
# training the Bagging estimator with undersampled training set
adaboost.fit(X_train_un, y_train_un)
```

#### Output:
```
AdaBoostClassifier(random_state=1)
```

### Code Block 118
```python
# Checking recall score on undersampled train and validation set
print(recall_score(y_train_un, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.6589458756795084
0.650649862150453
```

### Code Block 119
```python
# Confusion matrix for undersampled train data
cm = confusion_matrix(y_train_un, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 120
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_32.png)

### Code Block 121
```python
# training the Bagging estimator with undersampled training set
gboost.fit(X_train_un, y_train_un)
```

#### Output:
```
GradientBoostingClassifier(random_state=1)
```

### Code Block 122
```python
# Checking recall score on undersampled train and validation set
print(recall_score(y_train_un, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.6974710470337981
0.6612839700669555
```

### Code Block 123
```python
# Confusion matrix for undersampled train data
cm = confusion_matrix(y_train_un, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 124
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_33.png)

### Code Block 125
```python
# training the Bagging estimator with undersampled training set
xgboost.fit(X_train_un, y_train_un)
```

#### Output:
```
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=1, ...)
```

### Code Block 126
```python
# Checking recall score on undersampled train and validation set
print(recall_score(y_train_un, pred_train))
print(recall_score(y_val, pred_val))
```

#### Output:
```
0.8922240605057906
0.6703426545884207
```

### Code Block 127
```python
# Confusion matrix for undersampled train data
cm = confusion_matrix(y_train_un, pred_train)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
```

#### Output:
```
Text(58.222222222222214, 0.5, 'Actual Values')
```

### Code Block 128
```python
# Confusion matrix for validation data
cm = confusion_matrix(y_val, pred_val)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="g")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_34.png)

### Code Block 129
```python
%%time

# Choose the type of classifier.
rf2 = DecisionTreeClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 8, 10],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the random search
grid_obj = RandomizedSearchCV(rf2, parameters,n_iter=30, scoring=acc_scorer,cv=5, random_state = 1, n_jobs = -1, verbose = 2)
# using n_iter = 30, so randomized search will try 30 different combinations of hyperparameters
# by default, n_iter = 10

grid_obj = grid_obj.fit(X_train, y_train)

# Print the best combination of parameters
grid_obj.best_params_
```

#### Output:
```
Fitting 5 folds for each of 30 candidates, totalling 150 fits
CPU times: user 360 ms, sys: 94.1 ms, total: 454 ms
Wall time: 5.45 s
```

### Code Block 130
```python
grid_obj.best_score_
```

#### Output:
```
np.float64(0.5065077411305652)
```

### Code Block 131
```python
# Set the clf to the best combination of parameters
dt2_tuned = DecisionTreeClassifier(
 splitter= 'random',
 min_samples_split= 5,
 min_samples_leaf= 2,
 max_features= None,
 max_depth= 10,
 criterion= 'entropy'
)

# Fit the best algorithm to the data.
dt2_tuned.fit(X_train, y_train)
```

#### Output:
```
DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=2,
                       min_samples_split=5, splitter='random')
```

### Code Block 132
```python
# Checking recall score on train and validation set
print("Recall on train and validation set")
print(recall_score(y_train, dt2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(recall_score(y_val, dt2_tuned.fit(X_train, y_train)
.predict(X_val)))
print("")
print("Precision on train and validation set")
# Checking precision score on train and validation set
print(precision_score(y_train, dt2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(precision_score(y_val, dt2_tuned.fit(X_train, y_train)
.predict(X_val)))
print("")
print("Accuracy on train and validation set")
# Checking accuracy score on train and validation set
print(accuracy_score(y_train, dt2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(accuracy_score(y_val, dt2_tuned.fit(X_train, y_train)
.predict(X_val)))
```

#### Output:
```
Recall on train and validation set
0.5566060033089104
0.49940921622686096

Precision on train and validation set
0.6960945191992124
0.6384039900249376

Accuracy on train and validation set
0.7620094191522763
0.7373103087388801
```

### Code Block 133
```python
%%time

# Choose the type of classifier.
adaboost2 = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    'n_estimators': [50, 100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'estimator': [
        DecisionTreeClassifier(max_depth=1),  # Decision stump
        DecisionTreeClassifier(max_depth=2),  # Slightly deeper trees
        DecisionTreeClassifier(max_depth=3)   # Deeper trees
    ],
    'algorithm': ['SAMME', 'SAMME.R']
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the random search
grid_obj = RandomizedSearchCV(adaboost2, parameters,n_iter=30, scoring=acc_scorer,cv=5, random_state = 1, n_jobs = -1, verbose = 2)
# using n_iter = 30, so randomized search will try 30 different combinations of hyperparameters
# by default, n_iter = 10

grid_obj = grid_obj.fit(X_train, y_train)

# Print the best combination of parameters
grid_obj.best_params_
```

#### Output:
```
Fitting 5 folds for each of 30 candidates, totalling 150 fits
```

### Code Block 134
```python
grid_obj.best_score_
```

#### Output:
```
np.float64(0.4812147448511085)
```

### Code Block 135
```python
# Set the clf to the best combination of parameters
ada2_tuned = AdaBoostClassifier(
 n_estimators= 300,
 learning_rate= 1.0,
 estimator= DecisionTreeClassifier(max_depth=3),
 algorithm= 'SAMME'
)

# Fit the best algorithm to the data.
ada2_tuned.fit(X_train, y_train)
```

#### Output:
```
/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_weight_boosting.py:519: FutureWarning: The parameter 'algorithm' is deprecated in 1.6 and has no effect. It will be removed in version 1.8.
  warnings.warn(
```

### Code Block 136
```python
# Checking recall score on train and validation set
print("Recall on train and validation set")
print(recall_score(y_train, ada2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(recall_score(y_val, ada2_tuned.fit(X_train, y_train)
.predict(X_val)))
print("")
print("Precision on train and validation set")
# Checking precision score on train and validation set
print(precision_score(y_train, ada2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(precision_score(y_val, ada2_tuned.fit(X_train, y_train)
.predict(X_val)))
print("")
print("Accuracy on train and validation set")
# Checking accuracy score on train and validation set
print(accuracy_score(y_train, ada2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(accuracy_score(y_val, ada2_tuned.fit(X_train, y_train)
.predict(X_val)))
```

#### Output:
```
Recall on train and validation set
```

### Code Block 137
```python
%%time

# Choose the type of classifier.
gboost2 = GradientBoostingClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
    'n_estimators': [50, 100, 200, 300, 400],  # Number of boosting stages
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],  # Learning rate
    'max_depth': [3, 4, 5, 6, 7],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10, 20],  # Min samples to split a node
    'min_samples_leaf': [1, 2, 5, 10],  # Min samples at a leaf node
    'max_features': ['sqrt', 'log2', None, 0.1, 0.3, 0.5, 1],  # Max features for splitting nodes
    'subsample': [0.5, 0.7, 0.8, 1.0],  # Fraction of samples used for fitting each tree
    'loss': ['log_loss', 'exponential']  # Loss function to optimize
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the random search
grid_obj = RandomizedSearchCV(gboost2, parameters,n_iter=30, scoring=acc_scorer,cv=5, random_state = 1, n_jobs = -1, verbose = 2)
# using n_iter = 30, so randomized search will try 30 different combinations of hyperparameters
# by default, n_iter = 10

grid_obj = grid_obj.fit(X_train, y_train)

# Print the best combination of parameters
grid_obj.best_params_
```

#### Output:
```
Fitting 5 folds for each of 30 candidates, totalling 150 fits
CPU times: user 3.55 s, sys: 598 ms, total: 4.14 s
Wall time: 5min 30s
```

### Code Block 138
```python
grid_obj.best_score_
```

#### Output:
```
np.float64(0.4868885037163567)
```

### Code Block 139
```python
# Set the clf to the best combination of parameters
gboost2_tuned = GradientBoostingClassifier(
 subsample= 1.0,
 n_estimators= 50,
 min_samples_split= 20,
 min_samples_leaf= 2,
 max_features= 0.1,
 max_depth= 5,
 loss= 'log_loss',
 learning_rate=1.0
)

# Fit the best algorithm to the data.
gboost2_tuned.fit(X_train, y_train)
```

#### Output:
```
GradientBoostingClassifier(learning_rate=1.0, max_depth=5, max_features=0.1,
                           min_samples_leaf=2, min_samples_split=20,
                           n_estimators=50)
```

### Code Block 140
```python
# Checking recall score on train and validation set
print("Recall on train and validation set")
print(recall_score(y_train, gboost2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(recall_score(y_val, gboost2_tuned.fit(X_train, y_train)
.predict(X_val)))
print("")
print("Precision on train and validation set")
# Checking precision score on train and validation set
print(precision_score(y_train, gboost2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(precision_score(y_val, gboost2_tuned.fit(X_train, y_train)
.predict(X_val)))
print("")
print("Accuracy on train and validation set")
# Checking accuracy score on train and validation set
print(accuracy_score(y_train, gboost2_tuned.fit(X_train, y_train)
.predict(X_train)))
print(accuracy_score(y_val, gboost2_tuned.fit(X_train, y_train)
.predict(X_val)))
```

#### Output:
```
Recall on train and validation set
0.5613330181990073
0.47459629775502166

Precision on train and validation set
0.7097518753606463
0.6112214498510427

Accuracy on train and validation set
0.7783359497645211
0.7234432234432234
```

### Code Block 141
```python
model = dt2_tuned

# Checking recall score on test set
print("Recall on test set")
print(recall_score(y_test, model.predict(X_test)))
print("")

# Checking precision score on test set
print("Precision on test set")
print(precision_score(y_test, model.predict(X_test)))
print("")

# Checking accuracy score on test set
print("Accuracy on test set")
print(accuracy_score(y_test, model.predict(X_test)))
```

#### Output:
```
Recall on test set
0.4781323877068558

Precision on test set
0.6524193548387097

Accuracy on test set
0.7421507064364207
```

### Code Block 142
```python
model = ada2_tuned

# Checking recall score on test set
print("Recall on test set")
print(recall_score(y_test, model.predict(X_test)))
print("")

# Checking precision score on test set
print("Precision on test set")
print(precision_score(y_test, model.predict(X_test)))
print("")

# Checking accuracy score on test set
print("Accuracy on test set")
print(accuracy_score(y_test, model.predict(X_test)))
```

#### Output:
```
Recall on test set
0.516548463356974

Precision on test set
0.649331352154532

Accuracy on test set
0.7468602825745683
```

### Code Block 143
```python
model = gboost2_tuned

# Checking recall score on test set
print("Recall on test set")
print(recall_score(y_test, model.predict(X_test)))
print("")

# Checking precision score on test set
print("Precision on test set")
print(precision_score(y_test, model.predict(X_test)))
print("")

# Checking accuracy score on test set
print("Accuracy on test set")
print(accuracy_score(y_test, model.predict(X_test)))
```

#### Output:
```
Recall on test set
0.5135933806146572

Precision on test set
0.6145685997171145

Accuracy on test set
0.7315541601255887
```

### Code Block 144
```python
importances = ada2_tuned.feature_importances_
indices = np.argsort(importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

#### Output Image:
![Generated Plot](extracted_images\image_35.png)

