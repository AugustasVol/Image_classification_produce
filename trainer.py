#!/usr/bin/env python3
from cnn_keras import cnn
from images_dir_labels import list_paths_labels, image_iter
from pandas import get_dummies , read_csv
from sklearn.model_selection import train_test_split


epoch_num = 100
batch_size = 30
scale = 0.25
paths_train_X = "data_paths/X_train.csv"
paths_train_y = "data_paths/y_train.csv"

X_paths =read_csv(paths_train_X)
y =read_csv(paths_train_y)

# X from DataFrame to list
# y from DataFrame to numpy array
X_paths = list(X_paths[X_paths.columns[0]])
y = y.values

train_iter = image_iter(X_paths, y, batch_size = batch_size, scale=scale, normalize=False) 

input_shape = next(train_iter)[0][0].shape
output_num = len(y[0])

model_weights_path = "model_weights_" + str(scale)+"_"+str(output_num)
model = cnn(input_shape, output_num)
try:
    model.load_weights(model_weights_path)
except:
    pass
#manual feeding of data for training because not enough RAM
for _ in range(epoch_num):
    for batch_X, batch_y in train_iter:
        print(model.train_on_batch(batch_X,batch_y))
    model.save_weights(model_weights_path)
