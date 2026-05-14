import pandas as pd


def csv_to_bin(path_master_csv: str):
    """
    Data Processing:
        * With pneumonia: "Pneumonia" = 1
        * Normal: "No Findings" = 1
    """

    df = pd.read_csv(path_master_csv)

    df = df[(df["Pneumonia"] == 1) | (df["No Finding"] == 1)]

    df["target"] = 0
    df.loc[df["Pneumonia"] == 1, "target"] = 1

    df = df.rename(columns={
        "PngPath": "path_png",
        "PatientID": "patient_id"
    })

    df = df[["path_png", "patient_id", "target"]]

    df.to_csv("./dataset/dataset_binary.csv", index=False)