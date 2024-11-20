# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries, including `pandas` for data manipulation, `sklearn` for machine learning, and the Iris dataset from `sklearn.datasets`.

2. Load the Iris dataset using `load_iris()` from `sklearn.datasets`.

3. Create a DataFrame using `pd.DataFrame()` to organize the Iris dataset. Use `iris.data` for the features and `iris.target` for the target variable. Name the columns of the DataFrame using `iris.feature_names`.

4. Print the first few rows of the DataFrame using `df.head()` to verify the dataset's structure.

5. Separate the features (`X`) and the target variable (`y`). Drop the `target` column from the DataFrame to get `X`, and assign the `target` column to `y`.

6. Split the dataset into training and testing subsets using `train_test_split()`. Use `test_size=0.2` to allocate 20% of the data for testing and `random_state=42` for reproducibility.

7. Initialize an instance of `SGDClassifier` with parameters `max_iter=1000` and `tol=1e-3` to set the maximum number of iterations and the tolerance for stopping criteria.

8. Fit the `SGDClassifier` model to the training data (`X_train`, `y_train`) using the `fit()` method.

9. Predict the target values for the test data (`X_test`) using the trained model's `predict()` method. Store the predictions in `y_pred`.

10. Calculate the accuracy of the model using `accuracy_score()`, which compares the true target values (`y_test`) with the predicted values (`y_pred`).

11. Print the calculated accuracy, formatted to three decimal places.

12. Generate a confusion matrix using `confusion_matrix()` to evaluate the model's performance across all target classes.

13. Print the confusion matrix for analysis.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Dhivya Dharshini B 
RegisterNumber:  212223240031
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

```
## Output:
![Screenshot 2024-09-16 110403](https://github.com/user-attachments/assets/278ac232-c82e-451b-80f5-24c487bdb483)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
