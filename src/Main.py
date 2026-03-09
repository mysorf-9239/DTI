import os
import datetime
import argparse
import torch
import torch.nn as nn
from torch.utils import data
from loguru import logger
import numpy as np

from src.utils.engine import setup_seed, check_dir
from src.utils.config import load_config
from src.data.processor import prepare_dataloaders
from src.models.student.hdn import get_model
from src.models.teacher import SimpleMIDTI, build_midti_graphs
from src.models.fusion import UncertaintyGatedFusion
from src.core.trainer import Trainer
from DeepPurpose import utils as dp_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Deep_DTI: Student-Teacher Hybrid DTI Prediction")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    setup_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Using device: {device}")

    # Paths
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_root = os.path.join(base, cfg.output.root, cfg.dataset, timestamp)
    model_dir = os.path.join(base, cfg.output.root, "models")
    
    # Unified Data Pipeline
    (
        train_loader, valid_loader, test_loader, 
        nD, nP, dp_pairs, drug_id2local, prot_id2local
    ) = prepare_dataloaders(cfg.dataset, batch_size=cfg.train.batch_size)

    # Models
    # Student (Sequence Encoder)
    student = get_model(cfg.model.student).model.to(device)
    
    # Teacher (Graph features)
    dim = cfg.model.dim
    feat_drug = nn.Parameter(torch.randn(nD, dim, device=device) * 0.01)
    feat_prot = nn.Parameter(torch.randn(nP, dim, device=device) * 0.01)
    
    teacher = SimpleMIDTI(
        nD, nP, dim=dim, 
        n_heads=cfg.model.teacher.n_heads, 
        dia_layers=cfg.model.teacher.dia_layers, 
        dropout=cfg.model.teacher.dropout, 
        mlp_hidden=cfg.model.teacher.mlp_hidden
    ).to(device)
    
    # Fusion mechanism
    fusion = UncertaintyGatedFusion(
        student, teacher, 
        mc_samples=cfg.model.fusion.mc_samples, 
        temperature=cfg.model.fusion.temperature, 
        gate_hidden=cfg.model.fusion.gate_hidden
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(fusion.parameters()) + [feat_drug, feat_prot], 
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.train.lr_step_size, 
        gamma=cfg.train.lr_gamma
    )

    # Trainer
    trainer = Trainer(fusion, optimizer, scheduler, device, out_root, model_dir)

    def rebuild():
        return build_midti_graphs(
            feat_drug.detach().cpu().numpy(), 
            feat_prot.detach().cpu().numpy(), 
            dp_pairs, 
            k_dd=cfg.model.teacher.k_dd, 
            k_pp=cfg.model.teacher.k_pp, 
            device=device
        )

    # Training Loop
    best_auprc = -1.0
    best_ckpt = os.path.join(model_dir, f"teacher_gated_{cfg.dataset}_best.pt")
    patience, no_improve = cfg.train.patience, 0

    logger.info(f"Starting training pipeline for {cfg.dataset}...")
    for ep in range(1, cfg.train.epochs + 1):
        graphs = rebuild()
        avg_loss, auroc_train = trainer.train_epoch(ep, train_loader, graphs, feat_drug, feat_prot)
        val = trainer.evaluate(valid_loader, graphs, feat_drug, feat_prot)
        
        logger.info(f"Epoch {ep} | Loss: {avg_loss:.4f} | Val AUPRC: {val['auprc']:.4f}")

        if val["auprc"] > best_auprc:
            best_auprc = val["auprc"]
            no_improve = 0
            if cfg.output.save_model:
                torch.save({
                    "fusion": fusion.state_dict(), 
                    "feat_drug": feat_drug.detach().cpu(), 
                    "feat_prot": feat_prot.detach().cpu(),
                    "config": cfg
                }, best_ckpt)
                logger.info(f"✔ Best model saved at epoch {ep}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

    # Final Evaluation
    if os.path.exists(best_ckpt):
        payload = torch.load(best_ckpt, map_location=device)
        fusion.load_state_dict(payload["fusion"])
        feat_drug.data, feat_prot.data = payload["feat_drug"].to(device), payload["feat_prot"].to(device)
    
    trainer.export_test_csv(test_loader, rebuild(), feat_drug, feat_prot, os.path.join(out_root, "test_predictions.csv"))

if __name__ == "__main__":
    main()
