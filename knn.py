# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_x = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_x = test.values

#pca
pca = PCA(n_components=0.8)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

#knn
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(train_x, train_y)

#score
r = neigh.score(train_x,train_y)
print(r)

#save
test_y = neigh.predict(test_x)
pd.DataFrame({
    "ImageId": range(1, 1 + len(test_y)),
    "Label": test_y
}).to_csv(
    "knn_results.csv", header=True, index=False)
