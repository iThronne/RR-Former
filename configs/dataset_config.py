import pandas as pd
from pathlib import Path


class DatasetConfig:
    camels_root = Path("/data1/du/CAMELS/CAMELS-US/basin_dataset_public_v1p2")  # your CAMELS dataset root
    forcing_type = "daymet"  # TODO: "daymet" or "maurer_extended" or "nldas_extended"
    basin_mark = "673"  # TODO: daymet in [673, 671], maurer_extended in [448]
    basins_file = f"data/{basin_mark}basins_list.txt"
    global_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()

    # TODO: daymet date
    train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    test_start = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")

    # TODO: maurer_extended date
    # train_start = pd.to_datetime("2001-10-01", format="%Y-%m-%d")
    # train_end = pd.to_datetime("2008-09-30", format="%Y-%m-%d")
    # val_start = pd.to_datetime("1999-10-01", format="%Y-%m-%d")
    # val_end = pd.to_datetime("2001-09-30", format="%Y-%m-%d")
    # test_start = pd.to_datetime("1989-10-01", format="%Y-%m-%d")
    # test_end = pd.to_datetime("1999-09-30", format="%Y-%m-%d")

    dataset_info = f"{forcing_type}{basin_mark}_{train_start.year}~{train_end.year}#{val_start.year}~{val_end.year}#{test_start.year}~{test_end.year}"
