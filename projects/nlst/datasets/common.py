import pandas as pd


def get_dataset_split(split_csv):
    split_df = pd.read_csv(split_csv)
    split = {"train": [], "tune": [], "test": []}
    for split_name in split:
        for timepoint_name in ["T0", "T1", "T2"]:
            temp_df = split_df.loc[split_df.Data_Set == split_name.capitalize()]
            temp_df = temp_df.loc[temp_df[timepoint_name] == 1]
            scans = [f"{timepoint_name}/{id}_img.nrrd" for id in list(temp_df.Patient_ID)]
            split[split_name].extend(scans)
    return split