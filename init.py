import os

if __name__ == '__main__':
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists("files"):
        os.mkdir("files")
    if not os.path.exists("exp_results"):
        os.mkdir("exp_results")
        os.mkdir("exp_results/shells")
        os.mkdir("exp_results/data")
        os.mkdir("exp_results/logs")
    if not os.path.exists("dbconfig.py"):
        os.system("cp dbconfig_init.py dbconfig.py")
        print("Initialization successfully, the next step is to configure your database, please edit dbconfig.py to set your database configuration parameters.")