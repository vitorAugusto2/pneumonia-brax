# import pandas as pd
# from sklearn.model_selection import train_test_split
#
#
# def split_set(path_binary_csv: str, random_state: int = 42):
#     df = pd.read_csv(path_binary_csv)
#
#     # 1) Label por paciente
#     patient_df = (
#         df.groupby("patient_id", as_index=False)["target"]
#         .max()
#     )
#
#     # 2) Split por paciente
#     train_patients, temp_patients = train_test_split(
#         patient_df,
#         test_size=0.30,
#         stratify=patient_df["target"],
#         random_state=random_state
#     )
#
#     val_patients, test_patients = train_test_split(
#         temp_patients,
#         test_size=1/3,
#         stratify=temp_patients["target"],
#         random_state=random_state
#     )
#
#     # 3) Recupera imagens de cada split
#     train_df = df[df["patient_id"].isin(train_patients["patient_id"])].copy()
#     val_df = df[df["patient_id"].isin(val_patients["patient_id"])].copy()
#     test_df = df[df["patient_id"].isin(test_patients["patient_id"])].copy()
#
#     # 4) Salva
#     train_df.to_csv("./data/train.csv", index=False)
#     val_df.to_csv("./data/val.csv", index=False)
#     test_df.to_csv("./data/test.csv", index=False)
#
#     # 5) Prints
#     print("\n=> DISTRIBUIÇÃO DOS DADOS (PACIENTES)")
#     for name, d in [("TREINO", train_patients), ("VAL", val_patients), ("TESTE", test_patients)]:
#         counts = d["target"].value_counts().to_dict()
#         print(
#             f"{name}: Normal={counts.get(0, 0)} | "
#             f"Pneumonia={counts.get(1, 0)} | Total={len(d)}"
#         )
#
#     print("\n=> DISTRIBUIÇÃO DOS DADOS (IMAGENS)")
#     for name, d in [("TREINO", train_df), ("VAL", val_df), ("TESTE", test_df)]:
#         counts = d["target"].value_counts().to_dict()
#         print(
#             f"{name}: Normal={counts.get(0, 0)} | "
#             f"Pneumonia={counts.get(1, 0)} | Total={len(d)}"
#         )
#
#     return train_df, val_df, test_df

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from imblearn.under_sampling import RandomUnderSampler
#
#
# def split_set(path_binary_csv: str, sampling_strategy: float = 1, random_state: int = 42):
#     df = pd.read_csv(path_binary_csv)
#
#     # 1) Label por paciente
#     patient_df = (
#         df.groupby("patient_id", as_index=False)["target"]
#         .max()
#     )
#
#     # 2) Split por paciente
#     train_patients, temp_patients = train_test_split(
#         patient_df,
#         test_size=0.30,
#         stratify=patient_df["target"],
#         random_state=random_state
#     )
#
#     val_patients, test_patients = train_test_split(
#         temp_patients,
#         test_size=1/3,
#         stratify=temp_patients["target"],
#         random_state=random_state
#     )
#
#     # 3) Undersampling só no treino
#     rus = RandomUnderSampler(
#         sampling_strategy=sampling_strategy,
#         random_state=random_state
#     )
#
#     X_train_res, y_train_res = rus.fit_resample(
#         train_patients[["patient_id"]],
#         train_patients["target"]
#     )
#
#     train_patients = X_train_res.copy()
#     train_patients["target"] = y_train_res
#
#     # 4) Recupera imagens de cada split
#     train_df = df[df["patient_id"].isin(train_patients["patient_id"])].copy()
#     val_df = df[df["patient_id"].isin(val_patients["patient_id"])].copy()
#     test_df = df[df["patient_id"].isin(test_patients["patient_id"])].copy()
#
#     # 5) Salva
#     train_df.to_csv("./data/train.csv", index=False)
#     val_df.to_csv("./data/val.csv", index=False)
#     test_df.to_csv("./data/test.csv", index=False)
#
#     # 6) Prints
#     print("\n=> DISTRIBUIÇÃO DOS DADOS (PACIENTES)")
#     for name, d in [("TREINO", train_patients), ("VAL", val_patients), ("TESTE", test_patients)]:
#         counts = d["target"].value_counts().to_dict()
#         print(
#             f"{name}: Normal={counts.get(0, 0)} | "
#             f"Pneumonia={counts.get(1, 0)} | Total={len(d)}"
#         )
#
#     print("\n=> DISTRIBUIÇÃO DOS DADOS (IMAGENS)")
#     for name, d in [("TREINO", train_df), ("VAL", val_df), ("TESTE", test_df)]:
#         counts = d["target"].value_counts().to_dict()
#         print(
#             f"{name}: Normal={counts.get(0, 0)} | "
#             f"Pneumonia={counts.get(1, 0)} | Total={len(d)}"
#         )
#
#     return train_df, val_df, test_df



import pandas as pd
from sklearn.model_selection import train_test_split


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

    # Undersampling all
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

    # Prints: patients with or normal in each group
    print("\n=> DISTRIBUIÇÃO DOS DADOS (PACIENTES)")
    for name, d in [("TREINO", train_patients), ("VAL", val_patients), ("TESTE", test_patients)]:
        counts = d["target"].value_counts().to_dict()
        print(
            f"{name}: Normal={counts.get(0, 0)} | "
            f"Pneumonia={counts.get(1, 0)} | Total={len(d)}"
        )

    print("\n=> DISTRIBUIÇÃO DOS DADOS (IMAGENS)")
    for name, d in [("TREINO", train_df), ("VAL", val_df), ("TESTE", test_df)]:
        counts = d["target"].value_counts().to_dict()
        print(
            f"{name}: Normal={counts.get(0, 0)} | "
            f"Pneumonia={counts.get(1, 0)} | Total={len(d)}"
        )