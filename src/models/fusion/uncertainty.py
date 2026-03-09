import torch
import torch.nn as nn


def _mc_dropout_logits(forward_fn, n_samples: int = 6):
    logits = []
    for _ in range(int(n_samples)):
        logits.append(forward_fn())
    s = torch.stack(logits, dim=0)
    return s.mean(dim=0), s.var(dim=0, unbiased=False)


class PairGate(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, u_s: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        x = torch.stack([u_s, u_t], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class UncertaintyGatedFusion(nn.Module):
    def __init__(self, student_seq, teacher_midti, mc_samples=6, temperature=2.0, gate_hidden=32):
        super().__init__()
        self.student = student_seq
        self.teacher = teacher_midti
        self.mc_samples = int(mc_samples)
        self.T = float(temperature)
        self.gate = PairGate(hidden=int(gate_hidden))

    def forward(self, v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot, enable_mc=True):
        def student_fn(): return self.student(v_d, v_p).view(-1)
        def teacher_fn(): return self.teacher(graphs, feat_drug, feat_prot, d_idx, p_idx).view(-1)

        if enable_mc and self.mc_samples > 1:
            self.student.train(); self.teacher.train()
            logit_s, var_s = _mc_dropout_logits(student_fn, self.mc_samples)
            logit_t, var_t = _mc_dropout_logits(teacher_fn, self.mc_samples)
            u_s, u_t = var_s.detach(), var_t.detach()
        else:
            logit_s, logit_t = student_fn(), teacher_fn()
            u_s, u_t = torch.zeros_like(logit_s), torch.zeros_like(logit_t)

        w = self.gate(u_s, u_t)
        return w * logit_s + (1.0 - w) * logit_t, logit_s, logit_t, w, u_s, u_t

    def kd_loss(self, logit_s, logit_t):
        T = self.T
        ps, pt = torch.sigmoid(logit_s / T), torch.sigmoid(logit_t / T).detach()
        eps = 1e-7; ps, pt = ps.clamp(eps, 1 - eps), pt.clamp(eps, 1 - eps)
        kl = pt * torch.log(pt / ps) + (1 - pt) * torch.log((1 - pt) / (1 - ps))
        return kl.mean() * (T * T)
