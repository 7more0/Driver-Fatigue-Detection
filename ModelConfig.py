'''
    class ModelConfig for model configuration setting and storing
'''
import numpy as np


class ModelConfig:
    def __init__(self):
        self.data_augment = {
            'rotation_range': 30,
            'width_shift_range': 0.2,
            'height_shift_range': 0.3,
            'shear_range': 0.1,
            'zoom_range': 0.3,
            'horizontal_flip': True
        }       # data augment setting
        self.cnn_input_shape = (299, 299)       # input img shape of cnn model
        self.lstm_step = 30                     # lstm input node number
        self.cnn_frame_data_batchsize = 64      # frame batch size when training cnn

        self.labels = {'normal': np.array([1, 0]),
                       'yawning': np.array([0, 1])}
        self.feature_batchsize = 64     # clip batch size when training lstm
        self.test_sample_batch = 8

