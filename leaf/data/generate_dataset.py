import json
import os
import pickle

import numpy as np

dataset = "femnist"

train_folder = dataset + "/data/train"
test_folder = dataset + "/data/test"


if __name__ == '__main__':
    output_data = []
    validation_ratio = 0.1
    for file in os.listdir(train_folder):
        if file.endswith(".json"):
            f = open(os.path.join(train_folder, file), 'r')
            f2 = open(os.path.join(test_folder, file.replace("train", "test")), 'r')
            data = json.load(f)
            data2 = json.load(f2)
            for key in data["user_data"]:
                train_X = np.array(data["user_data"][key]["x"])
                train_y = np.array(data["user_data"][key]["y"])
                length = train_X.shape[0]
                idx = np.random.permutation(length)
                train_X = train_X[idx]
                train_y = train_y[idx]
                training_size = int(length * (1 - validation_ratio))
                validation_X = train_X[training_size:]
                validation_y = train_y[training_size:]
                train_X = train_X[:training_size]
                train_y = train_y[:training_size]
                test_X = np.array(data2["user_data"][key]["x"])
                test_y = np.array(data2["user_data"][key]["y"])
                output_data.append({
                    "train_X": train_X.reshape(-1, 1, 28, 28),
                    "train_y": train_y,
                    "validation_X": validation_X.reshape(-1, 1, 28, 28),
                    "validation_y": validation_y,
                    "test_X": test_X.reshape(-1, 1, 28, 28),
                    "test_y": test_y,
                })
                print(key, train_X.shape, validation_X.shape, test_X.shape)
    dataset = "femnist"
    split = "femnist"
    with open("../../files/" + dataset +"_"+ split +".pkl", 'wb') as fid:
        pickle.dump(output_data, fid)