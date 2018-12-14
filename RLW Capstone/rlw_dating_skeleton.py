"""
- at least two graphs containing exploration of the dataset
- a statement of your question (or questions!) and how you arrived there 
- the explanation of at least two new columns you created and how you did it
- the comparison between two classification approaches, including a qualitative
     discussion of simplicity, time to run the model, and accuracy, precision, and/or recall
- the comparison between two regression approaches, including a qualitative discussion of 
    simplicity, time to run the model, and accuracy, precision, and/or recall
- an overall conclusion, with a preliminary answer to your initial question(s), 
    next steps, and what other data you would like to have in order to better answer your question(s)
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import metrics
import time

# Create your df here:
df = pd.read_csv('profiles.csv')
# Unreported income removed
df = df.drop(df[df.income == -1].index)

#Explore data in df
#print(df.sign.value_counts())
#print()
#print(df.sex.value_counts())
#print()
#print(df.income.value_counts())
#print()
#print(df.pets.value_counts())
#print()
#print(df.orientation.value_counts())
#print()

"""
#Explore Plots
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.savefig('age.png')
plt.show()

plt.hist(df.sex, bins=2)
plt.xlabel("Sex")
plt.ylabel("Frequency")
plt.savefig('gender.png')
plt.show()

plt.hist(df.drinks, bins=7)
plt.xlabel("Drinks")
plt.ylabel("Frequency")
plt.savefig('drinks.png')
plt.show()
"""

#Map data in df to be more user friendly
sex_mapping = {"m": 0, "f": 1}
# Map income into 4 categories 20-40k, 50-100k 100-500k 500+k
income_mapping = {20000: 0, 100000: 3, 80000: 2, 30000: 0, 40000:0, 50000: 1, 60000: 1, 70000: 1, 150000: 3, 1000000: 4,
                  250000: 4, 500000: 4}
# Map pets to less categories: 
                 #0: dislikes pets
                 #1: likes pets does not have, 
                 #2: Like one not the other
                 #3: Owns one but dislikes other, 
                 #4: Owns one likes other, 
                 #5: owns 1, 
                 #6: has both                
pets_mapping = {"likes dogs and likes cats": 1,
                "likes dogs": 1,
                "likes dogs and has cats": 4,
                "has dogs": 5,
                "has dogs and likes cats": 4,
                "likes dogs and dislikes cats": 2,
                "has dogs and has cats": 6,
                "has cats": 5,
                "likes cats": 1,
                "has dogs and dislikes cats": 3,
                "dislikes dogs and likes cats": 2,
                "dislikes dogs and dislikes cats": 0,
                "dislikes cats": 0,
                "dislikes dogs and has cats": 3,
                "dislikes dogs": 0}

orientation_mapping = {"straight": 0, "gay": 1, "bisexual": 2}

# Map Data
df["sex_code"] = df.sex.map(sex_mapping)
df["income_code"] = df.income.map(income_mapping)
df["pets_code"] = df.pets.map(pets_mapping)
df["orientation_code"] = df.orientation.map(orientation_mapping)


df['sex_code'] = df['sex_code'].replace(np.nan, 0, regex=True)
df['income_code'] = df['income_code'].replace(np.nan, 0, regex=True)
df['pets_code'] = df['pets_code'].replace(np.nan, 0, regex=True)
df['orientation_code'] = df['orientation_code'].replace(np.nan, 0, regex=True)

# Print Data
#print(df.sex_code.value_counts())
#print(df.income_code.value_counts())
#print(df.pets_code.value_counts())
#print(df.orientation_code.value_counts())
#print()

feature_data = df[['sex_code', 'income_code', 'pets_code', 'orientation_code']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

print('Regression income given sex')
X = df[['sex_code']]
Y = df[['income_code']]
#plt.scatter(X, Y)
#plt.show()
x_train, x_test, y_train, y_test = train_test_split(X, Y)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
print(mlr.score(x_train, y_train))
print(mlr.coef_)
print(mlr.intercept_)
print()

print('Regression mlr income given orientation')
X = df[['orientation_code']]
Y = df[['income_code']]
# plt.scatter(X, Y)
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(X, Y)
mlr = LinearRegression()
start = time.time()
mlr.fit(x_train, y_train)
end = time.time()
print('mlr fit time: ', end - start)
print('mlr score', mlr.score(x_test, y_test))
y_predicted_mlr = mlr.predict(x_test)
# print('mlr metrics: ', metrics.classification_report(y_test, y_predicted_mlr))
print()

print('KN Regressor income given orientation')
regressor = KNeighborsRegressor(n_neighbors=3, weights='distance')
start = time.time()
regressor.fit(x_train, y_train)
end = time.time()
print('kn regressor fit time: ', end - start)
print('knr score', regressor.score(x_test, y_test))
y_predicted_knr = regressor.predict(x_test)
# print('knr metrics: ', metrics.classification_report(y_test, y_predicted_knr))
print()

# Guess if pets with income and orientaion
datapoints = df[['income_code', 'orientation_code']]
labels = df['pets_code']
# print(datapoints);
# plt.scatter(X, Y)
# plt.show()
x_train, x_test, y_train, y_test = train_test_split(datapoints, labels, random_state = 42)
print('KNeighborsClassifier')
knn = KNeighborsClassifier(n_neighbors=5)
start = time.time()
knn.fit(x_train, y_train)
end = time.time()
print('knn fit time: ', end - start)
y_predicted = knn.predict(x_test)
score = knn.score(x_test, y_test)
print('knn score: ', score)
print('knn metrics: ', metrics.classification_report(y_test, y_predicted))
print()

print('SVC classifier')
classifier = SVC(kernel='rbf', gamma=0.1)
start = time.time()
classifier.fit(x_train, y_train)
end = time.time()
print('svc fit time: ', end - start)
y_predicted_svc = classifier.predict(x_test)
score_svc = classifier.score(x_test, y_test)
print('SVC score: ', score_svc)
print('SVC metrics: ', metrics.classification_report(y_test, y_predicted_svc))
