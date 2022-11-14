import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import config
import pandas as pd
# lat400sqkm	lon400sqkm	gear	year	kg	     rate	   species	 sid
# Numirc        Numirc      Encode  Numirc  Numirc   Numirc    Encode    Encode
PICKLE_OBJ_PATH = config.PICKLE_OBJ_PATH
TRAINING = config.TRAINING
LABEL = config.LABEL
SCALE_LABEL = config.SCALE_LABEL
preprocess_map = config.PREPROCESS_DICT

class preprocess_fishiries():
    def __init__(self, data, shuffle_data=True, Training=TRAINING):
        self.data = data

        if shuffle_data:
            self.data.sample(frac=1).reset_index(drop=True)

        self.Training = Training
        self.fit_transform()

    def fit_transform(self):
        for item in preprocess_map.keys():
            operation = preprocess_map[item]
            if operation == "scale":
                if not SCALE_LABEL and item in [LABEL]:
                    continue
                self.data[item] = self.__scale(self.data[item], item)
            elif operation == "LabelEncoder":
                self.data[item] = self.__LabelEncoder(self.data[item], item)

    def __scale(self, data, col_name):
        path = os.path.join(PICKLE_OBJ_PATH, col_name+".pkl")
        if self.Training:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            pickle.dump(scaler, open(path, 'wb'))
        else:
            scaler = pickle.loads(path)
            scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
        return scaled_data

    def __LabelEncoder(self, data, col_name):
        path = os.path.join(PICKLE_OBJ_PATH, col_name+".pkl")
        if self.Training:
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data)
            pickle.dump(encoder, open(path, 'wb'))
        else:
            encoder = pickle.loads(path)
            encoded_data = encoder.transform(data)

        return encoded_data

    def split_x_y(self):
        self.y_data = self.data[LABEL]
        self.x_data = self.data.drop([LABEL], axis=1)
        return self.x_data, self.y_data

    def train_test_split(self, train_ratio=0.8):
        self.split_x_y()
        x_train_indx = int(train_ratio*len(self.x_data))
        self.x_train = self.x_data.iloc[:x_train_indx, :]

        if isinstance(LABEL,str):
            self.y_train = self.y_data.iloc[:x_train_indx]
            self.y_test = self.y_data.iloc[x_train_indx:]
        else:
            self.y_train = self.y_data.iloc[:x_train_indx, :]
            self.y_test = self.y_data.iloc[x_train_indx:, :]

        self.x_test = self.x_data.iloc[x_train_indx:, :]

        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_train_test_data(self):
        self.train_test_split()
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_data(self):
        return self.data
