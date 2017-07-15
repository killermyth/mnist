# -*- coding: utf-8 -*-
# !/usr/bin/env python

from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


class KNN(object):
    def __init__(self):
        self.train_file = "./csv/train.csv"

    def start(self):
        train_df = pd.read_csv(self.train_file)
        images = train_df.iloc[0:10000, 1:]
        labels = train_df.iloc[0:10000, 0]
        RadiusNeighborsClassifier
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,
                                                                                random_state=0)
        model = KNeighborsClassifier()
        model.fit(train_images, train_labels)
        predict = model.predict(test_images)
        scroe = accuracy_score(predict, test_labels)
        print scroe


if __name__ == '__main__':
    knn = KNN()
    knn.start()
