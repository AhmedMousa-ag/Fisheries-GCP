import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import config
# lat400sqkm	lon400sqkm	gear	year	kg	     rate	   species	 sid
# Numirc        Numirc      Encode  Numirc  Numirc   Numirc    Encode    Encode
PICKLE_OBJ_PATH = config.PICKLE_OBJ_PATH
PRODUCE_PICKLE_FILES = config.PRODUCE_PICKLE_FILES
preprocess_map = {"lat400sqkm": "scale",
                "lon400sqkm":"scale",
                "gear":"LabelEncoder",
                "year":"scale",
                "kg":"scale",
                "rate":"scale",
                "species":"LabelEncoder",
                "sid":"LabelEncoder"}

class preprocess_fishiries():
    def __init__(self,data,produce_pickle_files=PRODUCE_PICKLE_FILES):
        self.data = data
        self.produce_pickle_files = produce_pickle_files
        self.fit_transform()

    def fit_transform(self):
        for item in preprocess_map.keys():
            operation = preprocess_map[item]
            if operation == "scale":
                self.data[item] = self.__scale(self.data[item],item)
            elif operation == "LabelEncoder":
                self.data[item] = self.__LabelEncoder(self.data[item],item)
    
    def __scale(self,data,col_name):
        path = os.path.join(PICKLE_OBJ_PATH,col_name+".pkl")
        if self.produce_pickle_files:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1,1))
            pickle.dump(scaler,open(path, 'wb'))
        else:
            scaler = pickle.loads(path)
            scaled_data = scaler.transform(np.array(data).reshape(-1,1))
        return scaled_data

    def __LabelEncoder(self,data,col_name):
        path = os.path.join(PICKLE_OBJ_PATH,col_name+".pkl")
        if self.produce_pickle_files:
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data)
            pickle.dump(encoder,open(path, 'wb'))
        else:
            encoder = pickle.loads(path)
            encoded_data = encoder.transform(data)

        return encoded_data

    def get_data(self):
        return self.data