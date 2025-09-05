import torch
from torch import nn

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import init_weights

def Normalize(feat):
    return (feat - feat.mean(dim=0)) / (feat.std(dim=0, unbiased=False) + 1e-5)

def rsd_loss(feats_stu, feats_tea, kappa=0.01):
    B, D = feats_stu.shape[0], feats_stu.shape[1]
    feats_stu_norm, feats_tea_norm = Normalize(feats_stu), Normalize(feats_tea) # normalise along batch dim; [B,D]
    rcc = torch.mm(feats_tea_norm.T, feats_stu_norm) / B # representation cross-correlation matrix; [D,D]
    idt = torch.eye(D, device=rcc.device) # identity matrix [D,D]
    loss_rsd = (rcc - idt).pow(2) # information maximisation
    loss_rsd[(1 - idt).bool()] *= kappa # decorrelation
    return loss_rsd.sum()
    
@register_distiller
class RSD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(RSD, self).__init__(student, teacher, criterion, args)

        _, size_s = self.student.stage_info(self.args.rsd_stage)
        _, size_t = self.teacher.stage_info(self.args.rsd_stage)

        self.projector = nn.Sequential(
            nn.Linear(size_s, size_s * self.args.rsd_gamma, bias=False),
            nn.BatchNorm1d(size_s * self.args.rsd_gamma),
            nn.GELU(),
            nn.Linear(size_s * self.args.rsd_gamma, size_t, bias=False),
        )
        self.projector.apply(init_weights)

    def forward(self, image_weak, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            _, feat_teacher = self.teacher(image_weak, requires_feat=True)
        logits_student, feat_student = self.student(image_weak, requires_feat=True)

        loss_gt = self.criterion(logits_student, label)
        feat_t = feat_teacher[self.args.rsd_stage]
        feat_s_projected = self.projector(feat_student[self.args.rsd_stage])
        loss_rsd = self.args.rsd_loss_weight * rsd_loss(feat_s_projected, feat_t, self.args.rsd_kappa)

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_rsd": loss_rsd,
        }
        return logits_student, losses_dict
