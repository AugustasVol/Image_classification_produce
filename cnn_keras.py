import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def cnn(input_shape, output_number):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides = (2,2),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (3,3), strides = (3,3)))


    model.add(Conv2D(64, (3,3),strides = (2,2), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))
#    model.add(Conv2D(128, (3,3),strides = (2,2), activation = "relu"))
#    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3)))
#    model.add(Conv2D(256, (10,10), strides= (1,1), activation = "relu"))


    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_number, activation="softmax"))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
