from code.model.backbone import *
from code.config import *

# ----------------------------------
# Cosine Classifier with selectable init
# ----------------------------------

class CosineClassifier(nn.Module):
    """
    L2-normalised linear layer that returns s · cos(theta).

    Args
    ----
    in_dim        : feature dimension coming from the backbone
    num_classes   : starting number of classes (base session)
    init_method   : "kaiming" (default) | "l2"
                    - "kaiming": He normal then L2-norm
                    - "l2"     : sample from N(0,1) then L2-norm
    init_scale    : initial value for the scale/temperature s
    learnable_scale: if True, s is a learnable parameter
    """
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        *,
        init_method: str = "kaiming",
        init_scale: float = 10.0,
        learnable_scale: bool = True,
    ):
        super().__init__()

        w = torch.empty(num_classes, in_dim)

        if init_method.lower() == "kaiming":
            nn.init.kaiming_normal_(w)
        elif init_method.lower() == "l2":
            w.normal_()                       # N(0,1)
        else:
            raise ValueError(
                f'init_method must be "kaiming" or "l2", got {init_method}'
            )

        w = F.normalize(w, dim=1)             # ensure unit length
        self.weight = nn.Parameter(w)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        w = F.normalize(self.weight, dim=1)    # keep unit length in case of drift
        logits = self.scale * z @ w.t()    # (B, C)
        # return nn.functional.softmax(logits, dim=1)  # Softmax added for classification
        return logits  # for nn.crossentropy recieves logits

# ----------------------------------
# Encoder  (backbone + cosine FC)
# ----------------------------------
class Encoder(nn.Module):
    """
    Wrapper around ResNet-{12,18} with selectable weight-init for cosine FC.
    """
    def __init__(
        self,
        backbone_name: str = "resnet12",
        num_base_classes: int = NUM_CLASSES,
        feat_dim: int = 512,
        classifier_init: str = "l2",      # <-- choose "kaiming" or "l2"
        **classifier_kwargs,
    ):
        super().__init__()

        if backbone_name.lower() == "resnet12":
            self.backbone = ResNet12()
        elif backbone_name.lower() == "resnet18":
            self.backbone = ResNet18()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.classifier = CosineClassifier(
            in_dim=feat_dim,
            num_classes=num_base_classes,
            init_method=classifier_init,
            **classifier_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)

    @torch.no_grad()
    def add_weights(self, new_w):  # new_w:(N,D) already L2-normed
        self.classifier.weight.data = torch.cat([self.classifier.weight.data,
                                                 new_w.to(self.classifier.weight.device)], 0)


# if __name__ == "__main__":
#     model = Encoder(backbone_name="resnet12",
#                     num_base_classes=64,
#                     feat_dim=640)
#
#     # base training  …
#     logits = model(images)  # standard cross-entropy
#
#     # --- during an incremental session ---
#     with torch.no_grad():
#         feats = F.normalize(model.backbone(support_imgs), dim=1)
#     model.imprint(feats)  # add a new class on-the-fly


