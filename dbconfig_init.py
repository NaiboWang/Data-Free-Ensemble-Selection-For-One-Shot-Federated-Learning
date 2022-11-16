import os
import pymongo

# please modify the following configuration to your own mongodb server
myclient = pymongo.MongoClient('mongodb://username:password@ip:port/', connect=False)
mydb = myclient['exps']
# ensemble_selection_results = mydb["ensemble_selection_results"]
ensemble_selection_exp_results = mydb["ensemble_selection_exp_results"]
ensemble_selection_results = mydb["ensemble_selection_results_old2"]

def get_path():
    print("Use Local Data")
    DATASET_DIR = ""
    MODEL_DIR = ""
    return DATASET_DIR, MODEL_DIR