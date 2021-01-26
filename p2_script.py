#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow_addons import metrics

# Read the training dataset
df = pd.read_csv('./CE802_P2_Data/CE802_P2_Data.csv')

# Move the class column to the first column
klass = df['Class']
df = df.drop(columns=['Class'])
df.insert(0, 'Class', klass)

# Quick analysis of correlation of features and class
sns.set_theme()
# Compute the correlation matrix
corr_all = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_all, mask = mask, cmap = "BuPu")

# Preprocessing
def preprocess():
    df['Class'] = df['Class'].astype(int)
    x_cols = df.columns[1:]
    X, Y = df[x_cols], df['Class']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    return (x_train, y_train), (x_test, y_test)

# Get training and testing datasets
training, testing = preprocess()

# Test using DecisionTreeClassifier to be able to classify
clf = DecisionTreeClassifier(max_depth=8)
clf.fit(training[0], training[1])
clf.score(testing[0], testing[1])

# Test using Categorical Naive-Bayes to classify
clf = CategoricalNB()
clf.fit(training[0], training[1])
clf.score(testing[0], testing[1])

# Test using SVMs to classify
clf = SVC()
clf.fit(training[0], training[1])
clf.score(testing[0], testing[1])

# Use Neural Networks to classify
model = Sequential([
    Dense(10, activation='relu', input_shape=(len(df.columns[1:]),)),
    Dense(10, activation='tanh'),
    Dense(10, activation='sigmoid'),
    Dense(10, activation='tanh'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax'),
])
model.compile(optimizer='Adam', metrics=['accuracy', 'val_accuracy'])
history = model.fit(training[0], training[1], validation_data=testing, verbose=0).history
history[-1]['val_accuracy']
