from __future__ import annotations
from collections import defaultdict
import torch, torch.nn.functional as F

class SessionState:
    """Holds backbone, BiAG and running prototype/weight tables."""
    def __init__(self, backbone, cosine_classifier):
        self.backbone = backbone.eval().requires_grad_(False)
        self.classifier = cosine_classifier.eval().requires_grad_(False)
        self.biag     = None          # set after BiAG training
        self.protos   = []            # Tensor(C,D) later
        self.weights  = [cosine_classifier.weight.data]
        self.device   = next(backbone.parameters()).device

    @torch.no_grad()
    def extract_proto(self, imgs):
        z = self.backbone(imgs.to(self.device))
        return F.normalize(z.mean(0, keepdim=True), dim=1)

    def cat_weights(self, new_w):
        self.weights.append(new_w.detach())
