import pandas as pd
from sklearn.model_selection import train_test_split


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


def split_set(path_binary_csv: str):
    df = pd.read_csv(path_binary_csv)

    # Patient label
    patient_df = (
        df.groupby("patient_id")["target"]
        .max()
        .reset_index()
    )

    # Split patients
    patients_pnm = patient_df[patient_df["target"] == 1]
    patients_normal = patient_df[patient_df["target"] == 0]

    # Undersampling
    ratio = 1 # switch for 1 (1:1), 2 (1:2) or 3 (1:3) ...

    qt_pnm = len(patients_pnm)
    qt_normal = min(len(patients_normal), ratio * qt_pnm)

    patients_normal_sampled = patients_normal.sample(
        n = qt_normal,
        random_state = 42
    )

    patient_df_balanced = pd.concat(
        [patients_pnm, patients_normal_sampled],
        ignore_index = True
    )

    # Split stratified by patient
    train_patients, temp_patients = train_test_split(
        patient_df_balanced,
        test_size = 0.30,
        stratify = patient_df_balanced["target"],
        random_state = 42
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size = 1/3,
        stratify = temp_patients["target"],
        random_state = 42
    )

    # Recover images
    train_img = set(train_patients["patient_id"])
    val_img = set(val_patients["patient_id"])
    test_img = set(test_patients["patient_id"])

    train_df = df[df["patient_id"].isin(train_img)]
    val_df = df[df["patient_id"].isin(val_img)]
    test_df = df[df["patient_id"].isin(test_img)]

    train_df.to_csv("./data/train.csv", index=False)
    val_df.to_csv("./data/val.csv", index=False)
    test_df.to_csv("./data/test.csv", index=False)

    # Prints: patients with or without pneumonia in each group
    print(f"DISTRIBUTION OF PATIENTS BY GROUP")
    print(f"Pneumonia = {len(patients_pnm)}")
    print(f"Normal    = {len(patients_normal)}")

    print("\nTRAIN")
    print(train_df.groupby("target")["patient_id"].nunique().to_string())

    print("\nVAL")
    print(val_df.groupby("target")["patient_id"].nunique().to_string())

    print("\nTEST")
    print(test_df.groupby("target")["patient_id"].nunique().to_string())