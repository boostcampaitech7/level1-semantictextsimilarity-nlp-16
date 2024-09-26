import os
import pandas as pd
import glob
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def main(arg):
    dev_dir = os.path.join(arg.data_dir, "dev.csv")[['label']]
    dev = pd.read_csv(dev_dir)

    path = os.path.join(dev_dir, 'submission')
    all_files = glob.glob(os.path.join(path, "*.csv"))

    df_list = (pd.read_csv(file)[['target']] for file in all_files)
    df = pd.concat(df_list, axis=1)

    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2 = SVR(kernel='rbf')
    model3 = LinearRegression()

    model1.fit(df, dev)
    model2.fit(df, dev)
    model3.fit(df, dev)

    pred1 = model1.predict(df)
    pred2 = model2.predict(df)
    pred3 = model3.predict(df)
    
    pred1.to_csv("stacking_by_rf.csv")
    pred2.to_csv("stacking_by_svr.csv")
    pred3.to_csv("stacking_by_lin.csv")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d",
        "--data_dir",
        default=None,
        type=str,
        help="directory path for data (default: None)",
    )

    arg = args.parse_args()
    main(arg)