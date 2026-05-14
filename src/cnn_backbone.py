import copy
import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights
)


class HybridFineTuneModel(nn.Module):
    def __init__(self, cnn_name: str):
        super().__init__()
        self.cnn_name = cnn_name

        if cnn_name == "resnet50":
            self.weights = ResNet50_Weights.IMAGENET1K_V1
            backbone = resnet50(weights=self.weights)

            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()

            self.backbone = backbone
            self.head = nn.Linear(in_features, 1)
            self.feat_dim = in_features

        elif cnn_name == "vgg16":
            self.weights = VGG16_Weights.IMAGENET1K_V1
            backbone = vgg16(weights=self.weights)

            in_features = backbone.classifier[6].in_features
            backbone.classifier[6] = nn.Identity()

            self.backbone = backbone
            self.head = nn.Linear(in_features, 1)
            self.feat_dim = in_features

    def get_preprocess(self):
        return self.weights.transforms()

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(1)
        return logits

    def extract_features(self, x):
        return self.backbone(x)


def freeze_for_partial_finetune(model: HybridFineTuneModel):
    # congela tudo
    for param in model.backbone.parameters():
        param.requires_grad = False

    # descongela final
    if model.cnn_name == "resnet50":
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True

    elif model.cnn_name == "vgg16":
        for param in model.backbone.features[24:].parameters():
            param.requires_grad = True

    # cabeca treinavel
    for param in model.head.parameters():
        param.requires_grad = True

    return model


def train_partial_finetune(model, loaders, device, epochs, lr_backbone, lr_head):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith("backbone"):
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ])

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # treino
        model.train()
        train_loss = 0.0

        for imgs, ys in loaders["train"]:
            imgs = imgs.to(device)
            ys = ys.float().to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(loaders["train"].dataset)

        # validação
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for imgs, ys in loaders["val"]:
                imgs = imgs.to(device)
                ys = ys.float().to(device)

                logits = model(imgs)
                loss = criterion(logits, ys)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(loaders["val"].dataset)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def extract_features_from_finetuned(loader, model, device):
    model.eval()
    model.to(device)

    X_list = []
    y_list = []

    for imgs, ys in loader:
        imgs = imgs.to(device)
        feats = model.extract_features(imgs)

        X_list.append(feats.cpu())
        y_list.append(ys)

    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()

    return X, y


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device