# Soheb Osmani

# DataScience310Lab3

### Question 1:
An “ordinary least squares” (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.

True


### Question 2: 
Do you agree or disagree with the following statement: In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.


Disagree

### Question 3:
Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("L3Data.csv")
del df["questions"]

y = df['Grade'].values
X = df.loc[ : , (df.columns != 'Grade') ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

pred = lin_reg.predict(X_test)

print(metrics.mean_squared_error(y_test, pred))
69.29694824390998
q = 69.29694824390998
sqrt(q)
Answer: 8.3244 
```


### Question 4: 
In practice we determine the weights for linear regression with the "X_test" data.


False


### Question 5:

Polynomial regression is best suited for functional relationships that are non-linear in weights.

False

### Question 6:

Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

True

### Question 7:

Write your own code to import L3Data.csv into python as a data frame. Then save the feature values  'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("L3Data.csv")
del df["questions"]

y = df['Grade'].values
X = df.loc[ : , (df.columns != 'Grade') ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

print(len(X_train))
Answer: 23
```
### Question 8: 
The gradient descent method does not need any hyperparameters.


False

### Question 9: 

	
To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?

```
1st
import matplotlib.pyplot as plt

2nd
fig, ax = plt.subplots()

3rd
ax.scatter(X_test, y_test, color="black", label="Truth")
ax.scatter(X_test, lin_reg.predict(X_test), color="green", label="Linear")
ax.set_xlabel("Discussion Contributions")
ax.set_ylabel("Grade")

4th
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
```

### Question 10:

	
Which of the following forms is not  linear in the weights ?


![Screen Shot 2021-03-08 at 11 05 55 PM](https://user-images.githubusercontent.com/78623027/110417929-77355180-8064-11eb-810d-b68b96043d72.png)
