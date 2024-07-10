# wine_prediction
This project aims to predict the quality of red wine based on various chemical properties using a Random Forest Classifier.

## Table of Contents

1. [Importing the Dependencies](#importing-the-dependencies)
2. [Data Collection](#data-collection)
3. [Data Analysis and Visualization](#data-analysis-and-visualization)
4. [Correlation Analysis](#correlation-analysis)
5. [Data Preprocessing](#data-preprocessing)
6. [Label Binarization](#label-binarization)
7. [Train & Test Split](#train-test-split)
8. [Model Training](#model-training)
9. [Model Evaluation](#model-evaluation)
10. [Building a Predictive System](#building-a-predictive-system)

## Importing the Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

# loading the dataset to a Pandas DataFrame
```python
wine_dataset = pd.read_csv('/content/winequality-red.csv')
```
# number of rows & columns in the dataset
```python
print(wine_dataset.shape)
```

# first 5 rows of the dataset
```python
print(wine_dataset.head())
```

# checking for missing values
```python
print(wine_dataset.isnull().sum())
```
# statistical measures of the dataset
```python
print(wine_dataset.describe())
```
# number of values for each quality
```python
sns.catplot(x='quality', data=wine_dataset, kind='count')
```

# volatile acidity vs Quality
```python
plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
```

# citric acid vs Quality
```python
plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
```
# constructing a heatmap to understand the correlation between the columns
```python
correlation = wine_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
```
# separate the data and Label
```python
X = wine_dataset.drop('quality', axis=1)
print(X)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print(Y.shape, Y_train.shape, Y_test.shape)
model = RandomForestClassifier()
model.fit(X_train, Y_train)
```
# accuracy on test data
```python
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy : ', test_data_accuracy)
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
```

# changing the input data to a numpy array
```python
input_data_as_numpy_array = np.asarray(input_data)
```

# reshape the data as we are predicting the label for only one instance
```python
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
```

if (prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
