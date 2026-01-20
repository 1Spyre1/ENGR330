import pandas as pd
import numpy as np
import os
import xarray as xr
from utils import process_merged, load_era5, snap_to_025, create_sequences, compute_rh
from tqdm.auto import tqdm
import gc

    ## FIRE PATH READING
if True:
    fire_df = pd.read_csv("clean_fire.csv")

    fire_df["latitude"] = snap_to_025(fire_df["latitude"])
    fire_df["longitude"] = snap_to_025(fire_df["longitude"])

    fire_df["date"] = pd.to_datetime(fire_df["date"])
    fire_df["hour"] = fire_df["hour"].astype(int)

FOLDER_PATH = "era5_monthly"
i = 0
for dir in tqdm(os.listdir(FOLDER_PATH)):
    i+=1
    BASE_PATH = os.path.join(FOLDER_PATH, dir)

    try:

        ## ERA5 FILE PATHS
        era1 = f"era5_monthly/{dir}/{dir}.nc"
        if (not os.path.exists(era1)) or os.path.exists(os.path.join(BASE_PATH,"X.npy")):
            print("ERROR OR PASSED IN: ", BASE_PATH)
            continue
            

        #print(dir, BASE_PATH, era1, sep="|||")
        #break
        era5_df1 = load_era5(era1)


        ## MERGIN ERA5 and FIRE DATASETS

        merged = era5_df1.merge(
            fire_df[
                ["date", "hour", "latitude", "longitude", "fire"]
            ],
            on=["date", "hour", "latitude", "longitude"],
            how="left"
        )

        del era5_df1

        merged = process_merged(merged)

        merged = pd.get_dummies(merged, dtype=int, columns=["dominant_tree", "region"])

        merged["temperature"] , merged["humidity"] = compute_rh(merged["t2m"], merged["d2m"])
        merged["pressure"] = merged["sp"]/100

        print(merged.columns)
        feat = merged.columns.drop(["latitude", "longitude", "date", "hour", "t2m", "d2m", "sp", "fire"])

        ## INDEXING FOR TIME SERIES DATA
        X, y, t = create_sequences(merged, features= feat)
        del merged

        order = np.argsort(t)

        X = X[order]
        y = y[order]
        t = t[order]

        print(X.shape) 
        print(y.shape)
        print(t.shape)

        assert len(X) == len(y) == len(t)

        np.save(os.path.join(BASE_PATH,"X"), X, allow_pickle=True)
        np.save(os.path.join(BASE_PATH,"y"), y, allow_pickle=True)
        np.save(os.path.join(BASE_PATH,"t"), t, allow_pickle=True)

        del X, y, t, order
        gc.collect()
        print(i, ": ", dir, " : is completed...")

    except:
        continue

