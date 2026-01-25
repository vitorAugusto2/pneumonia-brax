import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

@torch.no_grad()
def extract_features(loader, model, device):
    model.to(device)

    X_list = []
    y_list = []

    for imgs, ys in loader:
        imgs = imgs.to(device)
        feats = model(imgs)
        X_list.append(feats.cpu())
        y_list.append(ys)

    X = torch.cat(X_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()

    return X, y