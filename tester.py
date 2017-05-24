from keras.models import load_model
import cnn_keras
from images_dir_labels import list_paths_labels, image_iter
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

model_weights_path = "model_weights_0.25_6"
batch_size = 10
paths_X_test = "data_paths/X_test.csv"
paths_y_test = "data_paths/y_test.csv"

scale = float(model_weights_path.split("_")[-2])

paths_X = read_csv(paths_X_test)
y = read_csv(paths_y_test)

y_names = list(map(lambda x: x.split("/")[-1], list(y.columns)))

paths_X = list(paths_X[paths_X.columns[0]])
y = y.values



test_iter = image_iter(paths_X, y, batch_size = batch_size, scale = scale, normalize=False)

shape, shape_output = next(test_iter)
shape = shape[0].shape
shape_output = shape_output.shape[1]


model = cnn_keras.cnn(shape, shape_output)
model.load_weights(model_weights_path)

#model.summary()

l = []
for batch_X,batch_y in test_iter:
#    plt.imshow(batch_X[0])
#    prediction = model.predict(batch_X)[0]
#    i = np.argmax(prediction)
#    print(y_names[i])
#    plt.show()
    
    o = model.test_on_batch(batch_X, batch_y)
    print(o)    
    l.append(o[1])
print(np.average(l))
