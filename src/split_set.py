import pandas as pd
from sklearn.model_selection import train_test_split


def csv_to_bin(path_master_csv: str):
    # Data Processing
    df = pd.read_csv(path_master_csv)

    df = df[df["Pneumonia"] != -1]

    df["target"] = (df["Pneumonia"] == 1).astype("int64")
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
    patients_pnm = patient_df[patient_df["target"] == 1]      # with pneumonia = 1
    patients_not_pnm = patient_df[patient_df["target"] == 0]  # without pneumonia = 0

    # Undersampling
    ratio = 1 # switch for 1 (1:1), 2 (1:2) or 3 (1:3) ...

    qt_pnm = len(patients_pnm)
    qt_not_pnm = min(len(patients_not_pnm), ratio * qt_pnm)

    patients_not_pnm_sampled = patients_not_pnm.sample(
        n = qt_not_pnm,
        random_state = 42
    )

    patient_df_balanced = pd.concat(
        [patients_pnm, patients_not_pnm_sampled],
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
    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    test_ids = set(test_patients["patient_id"])

    train_df = df[df["patient_id"].isin(train_ids)]
    val_df = df[df["patient_id"].isin(val_ids)]
    test_df = df[df["patient_id"].isin(test_ids)]

    train_df.to_csv("./data/train.csv", index=False)
    val_df.to_csv("./data/val.csv", index=False)
    test_df.to_csv("./data/test.csv", index=False)

    # Prints: patients with or without pneumonia in each group
    print(f"DISTRIBUTION OF PATIENTS BY GROUP")
    print(f"Patients with pneumonia      = {len(patients_pnm)}")
    print(f"Other diseases + No Findings = {len(patients_not_pnm)}")

    print("\nTRAIN")
    print(train_df.groupby("target")["patient_id"].nunique().to_string())

    print("\nVAL")
    print(val_df.groupby("target")["patient_id"].nunique().to_string())

    print("\nTEST")
    print(test_df.groupby("target")["patient_id"].nunique().to_string())