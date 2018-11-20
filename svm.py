# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

#get images and labels
labeld_images = pd.read_csv("train.csv")
images = labeld_images.iloc[0:5000,1:]
labels= labeld_images.iloc[0:5000,:1]

#preprocessing
images[images>0]=1
train_images,test_images,train_labels,test_labels = train_test_split(images,labels,train_size=0.8,random_state = 0)

# #show a number image
# i = 3
# img = images.iloc[i].as_matrix().reshape((28,28))
# plt.imshow(img,cmap="gray")
# plt.title(labels.iloc[i])
# #plt.hist(images.iloc[i])

#train and score
clf = svm.SVC()
clf.fit(train_images,train_labels.values.ravel())
r = clf.score(test_images,test_labels.values.ravel())
print(r)

# test
test_data = pd.read_csv("test.csv")
test_data[test_data>0]=1
results = clf.predict(test_data[0:5000])
print(results)

# save
df = pd.DataFrame(results)
df.index+=1 #index start from 1
df.index.name="imageId"
df.columns = ["Label"]
df.to_csv("svm_results.csv",header=True)