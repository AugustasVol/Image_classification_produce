#!/usr/bin/env python3

import images_dir_labels
import pandas as pd
from sklearn.model_selection import train_test_split
#directory where all pictures are
dir_path = # a directory where all data is located, diferent categories in diferent folders

path_X_train = "data_paths/X_train.csv"
path_X_test = "data_paths/X_test.csv"
path_y_train = "data_paths/y_train.csv"
path_y_test = "data_paths/y_test.csv"


X, y  = images_dir_labels.list_paths_labels(dir_path)
X = pd.DataFrame({"X_paths":X})
y = pd.get_dummies(y)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1)

X_train.to_csv(path_X_train, index = False)
X_test.to_csv(path_X_test, index = False)
y_train.to_csv(path_y_train, index = False)
y_test.to_csv(path_y_test, index = False)
