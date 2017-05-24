import os
import numpy as np
from skimage.io import imread_collection
from skimage.transform import resize, rescale

def list_paths_labels(label_dir):
    '''label_dir - a directory path
    for data, where
    directories act as a label,
    data in directories
    
    return paths_to_data_list, labels_list  '''
    dir_iter = os.walk(label_dir)
    next(dir_iter)

    X = []
    y = []
    for i in dir_iter:
        for file_name in i[2]:
            X.append(i[0]+ "/" + file_name)
            y.append(i[0])
    return X,y

class image_iter:
    ''' iterator to get batches of photos and labels
    initialize with a list of paths and labels'''
    def __init__(self, list_X, list_y, batch_size = 1, scale = 1, normalize = False):
        '''image_iter(X, Y)
            list_x - list of paths for photos
            list_y - list of labels
            batch_size
            scale - scale the image
            normalize - divide values by 255 to make the 0 to 1'''
 
        self.list_X = list_X
        self.list_y = list_y
        self.lenght = len(list_y)
        self.scale = scale
        self.iter_num = 0
        self.normalize = normalize
        self.batch_size = batch_size
    def __iter__(self):
        self.iter_num = 0
        return self
    def __next__(self):
        if self.iter_num >= self.lenght:
            raise StopIteration

        ret_batch_X =imread_collection(self.list_X[self.iter_num:self.iter_num + self.batch_size])
        ret_batch_X = list(map(lambda x: rescale(x, scale=self.scale), ret_batch_X))
        ret_batch_y =self.list_y[self.iter_num:self.iter_num + self.batch_size]
        self.iter_num = self.iter_num + self.batch_size
        ret_batch_X = np.array(ret_batch_X, dtype=np.float32)
        if self.normalize:
            ret_batch_X = np.divide(ret_batch_X, 255)
        return ret_batch_X, ret_batch_y
