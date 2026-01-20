import numpy as np
import xarray as xr
import pandas as pd
from tqdm.auto import tqdm

def process_merged(merged):

    merged = merged.dropna(
        subset=["t2m", "d2m", "sp"]
    ).reset_index(drop=True)

    merged["fire"] = merged["fire"].fillna(0).astype(int)

    merged = merged.astype({
        "fire": "int8",
        "hour": "int8",
        "latitude": "float64",
        "longitude": "float64",
        "t2m": "float32",
        "d2m": "float32",
        "sp": "float32"
    })

    conditions = [
        # 1. Pacific Northwest (Wet, dense conifers)
        (merged['latitude'] > 42) & (merged['longitude'] < -120),
        
        # 2. Southwest & California (Chaparral, Oak, Ponderosa)
        (merged['latitude'] <= 42) & (merged['longitude'] < -110),
        
        # 3. Rocky Mountains (High elevation, Spruce/Fir)
        (merged['latitude'] > 35) & (merged['longitude'].between(-120, -105)),
        
        # 4. Great Plains (Grasslands - lower tree density)
        (merged['longitude'].between(-105, -95)),
        
        # 5. Southeast (The Pine Belt - high fire frequency)
        (merged['latitude'] <= 37) & (merged['longitude'] > -95),
        
        # 6. Northeast & Midwest (Deciduous Hardwoods)
        (merged['latitude'] > 37) & (merged['longitude'] > -95)
    ]

    regions = [
        "Pacific Northwest", 
        "Southwest/CA", 
        "Rocky Mountains", 
        "Great Plains", 
        "Southeast", 
        "Northeast/Midwest"
    ]

    tree_types = [
        "Douglas Fir / Western Red Cedar",
        "Chaparral / Ponderosa Pine / Oak",
        "Engelmann Spruce / Subalpine Fir",
        "Grasslands / Eastern Redcedar",
        "Loblolly / Shortleaf Pine",
        "Maple / Beech / Birch / Oak"
    ]

    # Apply vectorization (Extremely fast)
    merged['region'] = np.select(conditions, regions, default="Other")
    merged['dominant_tree'] = np.select(conditions, tree_types, default="Mixed")

    return merged


def load_era5(era1):
    ds1 = xr.open_dataset(era1, engine="netcdf4")
    era5_df = (
        ds1
        .to_dataframe()
        .reset_index()
    )

    era5_df["date"] = era5_df["valid_time"].dt.date
    era5_df["hour"] = era5_df["valid_time"].dt.hour

    era5_df["latitude"] = snap_to_025(era5_df["latitude"])
    era5_df["longitude"] = snap_to_025(era5_df["longitude"])

    era5_df["date"] = pd.to_datetime(era5_df["date"])
    era5_df["hour"] = era5_df["hour"].astype(int)

    era5_df1 = era5_df.drop(columns=["number", "expver", "valid_time"])

    return era5_df1


def snap_to_025(x):
    return (np.round(x / 0.025) * 0.025).round(3)


def compute_rh(t, td):
    # t, td in Kelvin
    t_c  = t - 273.15
    td_c = td - 273.15
    rh = 100 * np.exp(
        (17.625 * td_c) / (243.04 + td_c)
        - (17.625 * t_c) / (243.04 + t_c)
    )
    return t_c, np.clip(rh, 0, 100)



def get_hard_negatives_optimized(fire_indices, ratio, lats, lons, labels, buffer=12):

    fire_indices = np.asarray(fire_indices)

    # 1. Create neighbor indices using broadcasting
    offsets = np.arange(-buffer, 0)
    potential_indices = fire_indices.reshape(-1, 1) + offsets

    # 2. Flatten & bounds check
    flat_indices = potential_indices.ravel()
    valid_mask = (flat_indices >= 0) & (flat_indices < len(labels))
    valid_indices = flat_indices[valid_mask]

    original_fire_indices = np.repeat(fire_indices, len(offsets))[valid_mask]

    loc_mask = (
        (lats[valid_indices] == lats[original_fire_indices]) &
        (lons[valid_indices] == lons[original_fire_indices])
    )

    candidate_indices = valid_indices[loc_mask]

    # 4. Must be negative
    hard_neg_pool = candidate_indices[labels[candidate_indices] == 0]

    # Remove duplicates
    hard_neg_pool = np.unique(hard_neg_pool)

    # 5. Safety guard
    if len(hard_neg_pool) == 0:
        return np.array([], dtype=int)

    num_to_sample = min(len(hard_neg_pool), len(fire_indices) * ratio)

    return np.random.choice(hard_neg_pool, size=num_to_sample, replace=False)




def create_sequences(df, features, seq_len=12, ratio=3,label_col='fire_next_h'):

    df = df.sort_values(['latitude', 'longitude', 'date', 'hour']).reset_index(drop=True)

    df[label_col] = (
        df.groupby(['latitude','longitude'])['fire']
        .shift(-1)
        .fillna(0)
        .astype(int)
    )
    # 1. Ensure data is sorted by location and time
    df['dt_combined'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
    df = df.sort_values(['latitude', 'longitude', 'dt_combined']).reset_index(drop=True)

    df_values = df[features].values
    lat_values = df['latitude'].values
    lon_values = df['longitude'].values
    time_values = df['dt_combined'].values
    labels = df[label_col].values
    
    X, y, t = [], [], []

    # 2. Separate fire and non-fire events
    fire_indices = df[df[label_col] == 1].index

    sampled_no_fire = get_hard_negatives_optimized(fire_indices, ratio, lat_values, lon_values, labels, seq_len)
    
    # Combine and shuffle targets
    target_indices = np.concatenate([fire_indices, sampled_no_fire])
    np.random.shuffle(target_indices)
    
    for idx in tqdm(target_indices):
            if idx >= seq_len:
                # Slicing numpy arrays is nearly instant
                lat_check = lat_values[idx - seq_len : idx]
                lon_check = lon_values[idx - seq_len : idx]
                
                # Validation
                if np.unique(lat_check).size == 1 and np.unique(lon_check).size == 1:
                    time_diff = (time_values[idx-1] - time_values[idx - (seq_len)]).astype('timedelta64[h]').astype(int)
                    
                    if time_diff == (seq_len-1):
                        X.append(df_values[idx - (seq_len) : idx])
                        y.append(labels[idx])
                        t.append(time_values[idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8), np.array(t)