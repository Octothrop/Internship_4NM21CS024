import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
names = ['class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']
data = pd.read_csv(url, names=names)

# EDA
print(data.head())
"""Output:
                  class    age menopause  ... breast breast_quad irradiat
0  no-recurrence-events  30-39   premeno  ...   left    left_low       no
1  no-recurrence-events  40-49   premeno  ...  right    right_up       no
2  no-recurrence-events  40-49   premeno  ...   left    left_low       no
3  no-recurrence-events  60-69      ge40  ...  right     left_up       no
4  no-recurrence-events  40-49   premeno  ...  right   right_low       no

[5 rows x 10 columns]"""
print(data.shape)
"""Output:
    (286, 10)"""
print(data.dtypes)
"""Output:
    class          object
    age            object
    menopause      object
    tumor_size     object
    inv_nodes      object
    node_caps      object
    deg_malig       int64
    breast         object
    breast_quad    object
    irradiat       object
    dtype: object"""
print(data.isnull().sum())
"""Output:
    class          0
    age            0
    menopause      0
    tumor_size     0
    inv_nodes      0
    node_caps      0
    deg_malig      0
    breast         0
    breast_quad    0
    irradiat       0
    dtype: int64"""
   
# Visualization of data
sns.countplot(x='class', data=data)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Distribution of Target Variable")
plt.show()

numerical_features = ['age', 'tumor_size', 'inv_nodes', 'deg_malig']
data[numerical_features].hist(bins=10, figsize=(10,8))
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Distribution of Numerical Features")
plt.show()

# relationship between the categorical features and the target variable
categorical_features = ['menopause', 'node_caps', 'breast', 'breast_quad', 'irradiat']
for feature in categorical_features:
    sns.countplot(x=feature, hue='class', data=data)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title("Distribution of " + feature + " by Class")
    plt.show()
   
# Preprocess the data
data = data.replace('?', 0)
data = pd.get_dummies(data, columns=['age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'breast', 'breast_quad', 'irradiat']) # One-hot encode categorical variables

# Spliting into features and target variable
X = data.drop(['class'], axis=1)
y = data['class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pg = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10, 100], 'solver': ['liblinear']}
gs = GridSearchCV(LogisticRegression(), pg, cv=5, verbose=2)
gs.fit(X_train, y_train)

# logistic regression model
lrm = gs.best_estimator_
lrm.fit(X_train, y_train)

# KNN classifier
knnm = KNeighborsClassifier(n_neighbors=5)
knnm.fit(X_train, y_train)

# Naive Bayes classifier
nm = GaussianNB()
nm.fit(X_train, y_train)

sc = lrm.score(X_test, y_test)
knns = knnm.score(X_test, y_test)
nbs = nm.score(X_test, y_test)

data = {'Classifier': ['Logistic Regression', 'KNN', 'Naive Bayes'],
        'Accuracy': [sc, knns, nbs]}
df = pd.DataFrame(data)
print(df)
"""Output:
            Classifier  Accuracy
0  Logistic Regression  0.706897
1                  KNN  0.689655
2          Naive Bayes  0.500000"""

# Saving the results in csv
#    results = pd.DataFrame({
#    'Classifier': ['Logistic Regression', 'KNN', 'Naive Bayes'],
#    'Accuracy': [score, knns, nbs]
#})
#results.to_csv('comparison_results.csv', index=False)
#print(results)"""

models = [
    ("LR", LogisticRegression()),
    ("KNN", KNeighborsClassifier()),
    ("NB", GaussianNB())
]
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')
print(f'Average accuracy: {np.mean(results):.4f}')
"""Output:
    LR: 0.6889 (0.0594)
    KNN: 0.6750 (0.0429)
    NB: 0.5672 (0.1559)
    Average accuracy: 0.6437"""

# Confusion matrix for the models
mod = [    ("LR", lrm),    ("KNN", knnm),    ("NB", nm)]
for name, model in mod :
    y_pred = model.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f"{name} - Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    # Display the plot
    plt.show()

#      THE DOCUMENTATION AND ALL GRAPHS ARE SEPERATELY UPLOADED AS A PDF FILE (P1_REGRESSION)        
