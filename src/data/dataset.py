import torch
from torch.utils import data
from DeepPurpose import utils as dp_utils

class DTI_Dataset(data.Dataset):
    """
    Custom Dataset for DTI prediction.
    Returns: v_d (drug encoding), v_p (protein embedding), y (label), 
             drug_local_idx, prot_local_idx
    """
    def __init__(self, df, drug_id2local, prot_id2local):
        self.df = df.reset_index(drop=True)
        self.drug_id2local = drug_id2local
        self.prot_id2local = prot_id2local

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        did = str(row.Graph_Drug)
        pid = row.Graph_Target

        d_idx = self.drug_id2local[did]
        p_idx = self.prot_id2local[pid]

        v_d = row.drug_encoding
        v_p = dp_utils.protein_2_embed(row.target_encoding)
        y = float(row.Seq_Label)
        return v_d, v_p, y, d_idx, p_idx
