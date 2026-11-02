import torch
from torch import nn

MODES = ("cnn", "transformer", "mean", "learned_gate", "reliability_gate")


class TinyCNN(nn.Module):
    """Small real CNN for offline pipeline tests, not the paper backbone."""
    num_features = 32

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class TinyTransformer(nn.Module):
    num_features = 32

    def __init__(self, image_size):
        super().__init__()
        if image_size % 8:
            raise ValueError("Tiny test transformer needs image_size divisible by 8")
        self.patch = nn.Conv2d(3, 32, 8, 8)
        self.pos = nn.Parameter(torch.zeros(1, (image_size // 8)**2, 32))
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        return self.norm(self.encoder(tokens + self.pos)).mean(1)


def backbone(name, pretrained, image_size, kind, weights=None):
    if name == "tiny_cnn":
        return TinyCNN()
    if name == "tiny_transformer":
        return TinyTransformer(image_size)
    import timm
    kwargs = {"img_size": image_size} if kind == "transformer" else {}
    if pretrained and weights:
        # A local timm/torchvision state dict replaces the Hugging Face download.
        kwargs["pretrained_cfg_overlay"] = {"file": str(weights)}
    return timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)


class FusionModel(nn.Module):
    def __init__(self, cnn="resnet18", transformer="vit_tiny_patch16_224", mode="reliability_gate",
                 pretrained=False, image_size=224, num_classes=3, gate_hidden=128,
                 cnn_weights=None, transformer_weights=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode}")
        self.mode = mode
        self.cnn = backbone(cnn, pretrained, image_size, "cnn", cnn_weights) if mode != "transformer" else None
        self.transformer = (backbone(transformer, pretrained, image_size, "transformer", transformer_weights)
                            if mode != "cnn" else None)
        if self.cnn is not None:
            self.cnn_head = nn.Linear(self.cnn.num_features, num_classes)
        if self.transformer is not None:
            self.transformer_head = nn.Linear(self.transformer.num_features, num_classes)
        if mode in ("learned_gate", "reliability_gate"):
            self.cnn_norm = nn.LayerNorm(self.cnn.num_features)
            self.transformer_norm = nn.LayerNorm(self.transformer.num_features)
            self.gate = nn.Sequential(nn.Linear(self.cnn.num_features + self.transformer.num_features,
                                                gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 2))

    def forward(self, x, uniform_gate=False):
        # No labels, augmented views, or oracle reliabilities enter inference.
        if self.mode == "cnn":
            logits = self.cnn_head(self.cnn(x))
            branch_logits = torch.stack([logits, logits], 1)
            weights = logits.new_tensor([1., 0.]).expand(x.shape[0], -1)
        elif self.mode == "transformer":
            logits = self.transformer_head(self.transformer(x))
            branch_logits = torch.stack([logits, logits], 1)
            weights = logits.new_tensor([0., 1.]).expand(x.shape[0], -1)
        else:
            cf, tf = self.cnn(x), self.transformer(x)
            branch_logits = torch.stack([self.cnn_head(cf), self.transformer_head(tf)], 1)
            weights = branch_logits.new_full((x.shape[0], 2), 0.5)
            if self.mode != "mean" and not uniform_gate:
                features = torch.cat([self.cnn_norm(cf), self.transformer_norm(tf)], -1)
                weights = self.gate(features).softmax(-1)
        probabilities = branch_logits.float().softmax(-1)
        fused = (weights.unsqueeze(-1) * probabilities).sum(1)
        return {"branch_logits": branch_logits, "weights": weights, "probs": fused}


def build_model(config, pretrained=None):
    names = config.get("class_names")
    return FusionModel(cnn=config["cnn"], transformer=config["transformer"], mode=config["mode"],
                       pretrained=config["pretrained"] if pretrained is None else pretrained,
                       image_size=config["image_size"], num_classes=3 if not names else len(names),
                       gate_hidden=config["gate_hidden"],
                       cnn_weights=config.get("cnn_weights"),
                       transformer_weights=config.get("transformer_weights"))
