import os
import numpy as np
import pandas as pd
from loguru import logger
from DeepPurpose import utils as dp_utils
from ..utils.engine import POS_LABEL, NEG_LABEL

def load_local_dataset(name: str):
    """Load dataset from CSV file in the data/ directory."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cands = [
        os.path.join(base, "data", f"{name}.csv"),
        os.path.join(base, "data", f"{name.lower()}.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"CSV file not found for dataset: {name}")

def make_binary_labels(df: pd.DataFrame, name: str):
    """Binarize continuous labels based on dataset-specific thresholds."""
    df = df.copy()
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)
        return df

    if "Y" not in df.columns:
        raise ValueError("CSV must contain 'Label' or 'Y' column.")

    df = df.dropna(subset=["Drug_ID", "Target_ID", "Drug", "Target", "Y"])
    y = pd.to_numeric(df["Y"], errors="coerce").values.astype(float)

    if name.upper() == "DAVIS":
        pY = -np.log10(y * 1e-9 + 1e-12)
        df["pY"] = pY
        df["Label"] = (df["pY"] >= 7.0).astype(int)
    elif name == "BindingDB_Kd":
        pY = -np.log10(y * 1e-9 + 1e-12)
        df["pY"] = pY
        df["Label"] = (df["pY"] >= 7.6).astype(int)
    elif name.upper() == "KIBA":
        df["Label"] = (y >= 12.1).astype(int)
    else:
        raise ValueError(f"Unsupported dataset for binarization: {name}")
    return df

def sample_stat(df: pd.DataFrame):
    """Print statistics of negative/positive samples."""
    neg_n = int((df["Label"] == NEG_LABEL).sum())
    pos_n = int((df["Label"] == POS_LABEL).sum())
    logger.info(f"neg/pos = {neg_n}/{pos_n} | neg%={100*neg_n/max(1,neg_n+pos_n):.2f}%")
    return neg_n, pos_n

def df_data_preprocess(df: pd.DataFrame, undersampling: bool = True):
    """Clean and optionally undersample the dataset to balance classes."""
    df = df.dropna().copy()
    df["Drug_ID"] = df["Drug_ID"].astype(str)
    neg_n, pos_n = sample_stat(df)

    if undersampling:
        neg_df_all = df[df["Label"] == NEG_LABEL]
        if len(neg_df_all) > pos_n:
            neg_df = neg_df_all.sample(n=pos_n, random_state=1)
        else:
            neg_df = neg_df_all
        pos_df = df[df["Label"] == POS_LABEL]
        df = pd.concat([pos_df, neg_df], ignore_index=True)
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    sample_stat(df)
    return df

def df_data_split(df: pd.DataFrame, frac: tuple = (0.7, 0.1, 0.2)):
    """Split the dataframe into train, validation, and test sets."""
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    total = len(df)
    t1 = int(total * frac[0])
    t2 = int(total * (frac[0] + frac[1]))

    train = df.iloc[:t1].copy()
    valid = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()

    logger.info("Train stats:")
    sample_stat(train)
    logger.info("Valid stats:")
    sample_stat(valid)
    logger.info("Test stats:")
    sample_stat(test)
    return train, valid, test

def dti_df_process(df: pd.DataFrame):
    """Encode drug and protein sequences using DeepPurpose."""
    df2 = pd.DataFrame({
        "Seq_Drug": df["Drug"].values,
        "Seq_Target": df["Target"].values,
        "Seq_Label": df["Label"].values,
        "Graph_Drug": df["Drug_ID"].astype(str).values,
        "Graph_Target": df["Target_ID"].values
    })
    df2 = dp_utils.encode_drug(df2, "MPNN", column_name="Seq_Drug")
    df2 = dp_utils.encode_protein(df2, "CNN", column_name="Seq_Target")
    return df2

def prepare_dataloaders(dataset_name: str, batch_size: int = 32):
    """
    Unified entry point for the entire data pipeline.
    Returns: (train_loader, valid_loader, test_loader, nD, nP, dp_pairs, drug_id2local, prot_id2local)
    """
    from .dataset import DTI_Dataset
    from torch.utils import data

    logger.info(f"Preparing unified dataloaders for: {dataset_name}")
    
    # 1. Load and Clean
    df_raw = load_local_dataset(dataset_name)
    df_raw = make_binary_labels(df_raw, dataset_name)
    df_raw = df_data_preprocess(df_raw, undersampling=True)
    
    # 2. Split
    train_df, valid_df, test_df = df_data_split(df_raw)
    
    # 3. DeepPurpose Encoding
    train_df = dti_df_process(train_df)
    valid_df = dti_df_process(valid_df)
    test_df  = dti_df_process(test_df)

    # 4. ID Mapping for Graph models
    drug_ids = df_raw["Drug_ID"].astype(str).unique().tolist()
    prot_ids = df_raw["Target_ID"].unique().tolist()
    drug_id2local = {d: i for i, d in enumerate(drug_ids)}
    prot_id2local = {p: i for i, p in enumerate(prot_ids)}
    nD, nP = len(drug_ids), len(prot_ids)

    # 5. Graph edges
    # By default we only use TRAIN interactions to build the D-P bipartite graph.
    # If you include valid/test interactions here, validation metrics can be inflated
    # due to information leakage (the evaluated pairs become explicit graph edges).
    dp_pairs_mode = os.environ.get("UGTS_DTI_DP_PAIRS", "train").strip().lower()
    pairs_df = train_df
    if dp_pairs_mode in {"all", "full"}:
        logger.warning("UGTS_DTI_DP_PAIRS=all enabled: this leaks valid/test pairs into the graph.")
        pairs_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    dp_pairs = np.stack(
        [
            pairs_df["Graph_Drug"].astype(str).map(drug_id2local).values,
            pairs_df["Graph_Target"].map(prot_id2local).values,
        ],
        axis=1,
    )

    # 6. Dataloaders
    train_loader = data.DataLoader(
        DTI_Dataset(train_df, drug_id2local, prot_id2local), 
        batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=dp_utils.mpnn_collate_func
    )
    valid_loader = data.DataLoader(
        DTI_Dataset(valid_df, drug_id2local, prot_id2local), 
        batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dp_utils.mpnn_collate_func
    )
    test_loader = data.DataLoader(
        DTI_Dataset(test_df, drug_id2local, prot_id2local), 
        batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dp_utils.mpnn_collate_func
    )

    return train_loader, valid_loader, test_loader, nD, nP, dp_pairs, drug_id2local, prot_id2local
