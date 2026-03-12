import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from ..utils.engine import check_dir, csv_record, to_device


class Trainer:
    """
    Trainer class to handle training, validation, evaluation, and exporting results.
    """

    def __init__(self, model, optimizer, scheduler, device, out_root, model_dir):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_root = out_root
        self.model_dir = model_dir
        check_dir(out_root)
        check_dir(model_dir)

    def train_epoch(self, epoch, loader, graphs, feat_drug, feat_prot):
        self.model.train()
        total_loss = 0.0
        y_score_train, y_true_train = [], []

        for bi, (v_d, v_p, y, d_idx, p_idx) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            self.optimizer.zero_grad()

            v_d = to_device(v_d, self.device)
            v_p = to_device(v_p, self.device)
            d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=self.device)
            p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

            logit, logit_s, logit_t, w, u_s, u_t = self.model(
                v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot, enable_mc=True
            )

            L_sup = F.binary_cross_entropy_with_logits(logit.view(-1), y)
            L_kd = self.model.kd_loss(logit_s.view(-1), logit_t.view(-1))
            L_reg = ((w - 0.5) ** 2).mean()

            loss = L_sup + 0.1 * L_kd + 0.01 * L_reg
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            y_true_train.append(y.detach().cpu().numpy())
            y_score_train.append(logit.detach().cpu().numpy())

            csv_record(
                os.path.join(self.out_root, "loss.csv"),
                {
                    "epoch": epoch,
                    "batch": bi,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "loss": float(loss.item()),
                    "avg_loss": total_loss / (bi + 1),
                },
            )

        self.scheduler.step()
        y_true_train = np.concatenate(y_true_train)
        y_score_train = np.concatenate(y_score_train)
        auroc_train = roc_auc_score(y_true_train, y_score_train)
        return total_loss / len(loader), auroc_train

    @torch.no_grad()
    def evaluate(self, loader, graphs, feat_drug, feat_prot):
        self.model.eval()
        y_true = []
        y_score_f, y_score_s, y_score_t = [], [], []
        w_all = []

        for v_d, v_p, y, d_idx, p_idx in tqdm(loader, desc="Evaluating"):
            v_d = to_device(v_d, self.device)
            v_p = to_device(v_p, self.device)
            d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=self.device)
            p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

            # The fusion gate is driven by uncertainty (MC-dropout variance). If we disable
            # MC at inference time, u_s/u_t become zeros and the gate collapses to a constant.
            logit_f, logit_s, logit_t, w, _, _ = self.model(
                v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot, enable_mc=True
            )
            y_true.append(y.detach().cpu().numpy())
            y_score_f.append(logit_f.detach().cpu().numpy())
            y_score_s.append(logit_s.detach().cpu().numpy())
            y_score_t.append(logit_t.detach().cpu().numpy())
            w_all.append(w.detach().cpu().numpy())

        y_true = np.concatenate(y_true)
        y_score_f = np.concatenate(y_score_f)
        y_score_s = np.concatenate(y_score_s)
        y_score_t = np.concatenate(y_score_t)
        w_all = np.concatenate(w_all)
        prob_f = 1.0 / (1.0 + np.exp(-y_score_f))
        prob_s = 1.0 / (1.0 + np.exp(-y_score_s))
        prob_t = 1.0 / (1.0 + np.exp(-y_score_t))

        from ..utils.metrics import all_dti_metrics

        m = all_dti_metrics(y_true, prob_f)
        m["auroc"] = float(roc_auc_score(y_true, prob_f))
        m["auprc"] = float(average_precision_score(y_true, prob_f))
        m["gate_mean"] = float(np.mean(w_all))
        m["auroc_student"] = float(roc_auc_score(y_true, prob_s))
        m["auprc_student"] = float(average_precision_score(y_true, prob_s))
        m["auroc_teacher"] = float(roc_auc_score(y_true, prob_t))
        m["auprc_teacher"] = float(average_precision_score(y_true, prob_t))
        return m

    @torch.no_grad()
    def export_test_csv(self, loader, graphs, feat_drug, feat_prot, save_path):
        self.model.eval()
        rows = []

        for v_d, v_p, y, d_idx, p_idx in tqdm(loader, desc="Exporting CSV"):
            v_d = to_device(v_d, self.device)
            v_p = to_device(v_p, self.device)
            d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=self.device)
            p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.float32, device=self.device)

            logit, logit_s, logit_t, w, u_s, u_t = self.model(
                v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot, enable_mc=True
            )

            prob_f = torch.sigmoid(logit)
            prob_s = torch.sigmoid(logit_s)
            prob_t = torch.sigmoid(logit_t)

            for i in range(len(y)):
                rows.append(
                    {
                        "Drug_ID": int(d_idx[i].cpu()),
                        "Target_ID": int(p_idx[i].cpu()),
                        "Label": int(y[i].cpu()),
                        "Student": float(prob_s[i].cpu()),
                        "Teacher": float(prob_t[i].cpu()),
                        "Fusion": float(prob_f[i].cpu()),
                    }
                )

        pd.DataFrame(rows).to_csv(save_path, index=False)
        logger.info(f"Test CSV saved to {save_path}")
