# -*- coding: utf-8 -*-
# !/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')


class SoftMax(object):
    def __init__(self):
        self.train_file = "./csv/train.csv"
        self.test_file = "./csv/test.csv"

    def start(self):
        train_df = pd.read_csv(self.train_file)
        images = train_df.iloc[0:10000, 1:]
        labels = train_df.iloc[0:10000, 0]
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8,
                                                                                random_state=0)
        model = LogisticRegression().fit(train_images, train_labels)
        predict = model.predict(test_images)
        scroe = accuracy_score(predict, test_labels)
        print scroe

    def show_img(self, series):
        first_data = series.reshape((28, 28))
        plt.imshow(first_data, cmap='gray')
        plt.show()


if __name__ == '__main__':
    softMax = SoftMax()
    softMax.start()
