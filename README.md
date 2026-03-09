# DeepDTI: Uncertainty-Gated Student-Teacher Learning for Drug-Target Interaction Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

DeepDTI is a hybrid deep learning framework designed for robust Drug-Target Interaction (DTI) prediction. It leverages a **Student-Teacher architecture** combining sequence-based deep learning (Student) with graph-based interactive attention networks (Teacher), fused dynamically using **Uncertainty-Gated Fusion**.

---

## 🌟 Key Features

- **Hybrid Architecture**: Combines **DeepPurpose** (Sequence-based) and **MIDTI** (Graph-based) paradigms.
- **Uncertainty-Aware Fusion**: Implements **MC Dropout** to estimate prediction uncertainty, allowing the model to dynamically weight Student and Teacher outputs via a `PairGate`.
- **Knowledge Distillation**: Features a binary KD loss to guide the Student model using Teacher representations.
- **Advanced Graph Learning**: Utilizes **Deep Interactive Attention (DIA)** blocks and multi-view graph convolution (DD, PP, DP, DDPP).
- **Researcher-Friendly CLI**: Fully modularized codebase with YAML-based configuration management.
- **Comprehensive Metrics**: Optimized for research with AUPRC, AUROC, Concordance Index (CI), Pearson correlation, and more.

---

## 🏗 Architecture Overview

DeepDTI operates on a dual-pathway mechanism:

1.  **Student (Sequence Path)**: Uses Message Passing Neural Networks (MPNN) for drug SMILES and Convolutional Neural Networks (CNN) for protein sequences.
2.  **Teacher (Graph Path)**: Constructs k-Nearest Neighbor (kNN) similarity graphs and applies interactive attention to capture deep structural relationships.
3.  **Fusion Layer**: Calculates uncertainty $(\sigma^2)$ for both paths and uses a gating network to compute a dynamic weight $w$.

### Data & Model Flow

```mermaid
graph TB
    subgraph Data_Inputs ["1. Input Representation"]
        D_SMILES["Drug SMILES"]
        P_FASTA["Protein FASTA"]
        G_kNN["kNN Interaction Graphs<br/>(DD, PP, DP, DDPP)"]
    end

    subgraph Student_Branch ["2. Student Branch (Sequence-based)"]
        MPNN["MPNN Encoder<br/>(Message Passing)"]
        CNN["CNN Encoder<br/>(Local Patterns)"]
        S_MLP["Student Predictor"]
        D_SMILES --> MPNN
        P_FASTA --> CNN
        MPNN --> S_MLP
        CNN --> S_MLP
    end

    subgraph Teacher_Branch ["3. Teacher Branch (Graph-based)"]
        G_Feat["Drug/Prot Graph Features"]
        GCN["Graph Convolutional Stack<br/>(Multi-view)"]
        DIA["DIA Blocks<br/>(Deep Interactive Attention)"]
        T_MLP["Teacher Predictor"]
        
        G_kNN --> GCN
        G_Feat --> GCN
        GCN --> DIA
        DIA --> T_MLP
    end

    subgraph Uncertainty_Module ["4. Uncertainty & Gated Fusion"]
        direction TB
        MC_S["MC Dropout samples (S)"]
        MC_T["MC Dropout samples (T)"]
        
        Var_S["Uncertainty (σ²_S)"]
        Var_T["Uncertainty (σ²_T)"]
        
        PG["PairGate Network"]
        Weight["Gate Weight (w)"]
        
        S_MLP -- "S samples" --> MC_S --> Var_S --> PG
        T_MLP -- "S samples" --> MC_T --> Var_T --> PG
        PG --> Weight
    end

    subgraph Output_Loss ["5. Final Layer & Training Losses"]
        direction LR
        Final_Logit["Fused Prediction"]
        BCE_Loss["BCE Supervised Loss"]
        KD_Loss["Knowledge Distillation Loss<br/>(T -> S)"]
        
        Weight -- "w*S + (1-w)*T" --> Final_Logit
        Final_Logit --> BCE_Loss
        S_MLP -. "Distillation" .-> KD_Loss
        T_MLP -. "Distillation" .-> KD_Loss
    end

    %% Styling
    style Data_Inputs fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Student_Branch fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Teacher_Branch fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Uncertainty_Module fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style Output_Loss fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
```

---

## 📂 Project Structure

```text
src/
├── data/           # Unified data loading and binarization
├── models/         # Advanced architectures
│   ├── student/    # Sequence-based encoders
│   ├── teacher/    # Graph-based DIA networks
│   └── fusion/     # Uncertainty-gated integration logic
├── core/           # Standardized Training/Evaluation engine
├── utils/          # Config loaders, metrics, and reproducibility
└── main.py         # Primary entry point
configs/            # Experiment configuration (YAML)
data/               # Datasets (DAVIS, KIBA, BindingDB)
output/             # Results, checkpoints, and logs
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/your-repo/DeepDTI.git
cd DeepDTI
pip install -r requirements.txt
```

### 2. Configuration

All hyperparameters are managed via `configs/default.yaml`. You can modify model dimensions, training epochs, and dataset paths directly in this file.

### 3. Running Experiments

```bash
# Run with default configuration (KIBA)
python -m src.main

# Run with a specific experiment config
python -m src.main --config configs/davis_experiment.yaml
```

---

## 📊 Evaluation & Methodology

DeepDTI supports a wide array of metrics to ensure state-of-the-art evaluation:

| Category | Metrics |
| :--- | :--- |
| **Traditional DTI** | AUPRC, AUROC |
| **Regression-style** | Concordance Index (CI), MSE, RMSE, Pearson |
| **Classification** | Accuracy, F1, Precision, Recall, Specificity |

### Training Logic
The framework uses a composite loss function:
$$ \mathcal{L} = \mathcal{L}_{BCE} + \alpha \cdot \mathcal{L}_{KD} + \beta \cdot \mathcal{L}_{Reg} $$
Where $\mathcal{L}_{KD}$ is the binary knowledge distillation loss and $\mathcal{L}_{Reg}$ is the gate regularization.

---

## 📝 Citation

If you use this code in your research, please cite our project:

```bibtex
@article{deepdti2026,
  title={DeepDTI: Uncertainty-Gated Student-Teacher Learning for Drug-Target Interaction Prediction},
  author={DeepDTI Team},
  year={2026},
  journal={arXiv preprint}
}
```

---

## ⚖ License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
