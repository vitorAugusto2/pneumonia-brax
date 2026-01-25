import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights
)


def get_backbone(cnn_name: str):
    if cnn_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1    # database IMAGENET
        preprocess = weights.transforms()           # pre-processing transformations applied to input data
        model = resnet50(weights=weights)           # CNN ResNet50
        model.fc = nn.Identity()                    # remove last layer -> [out] (B, 2048)
        feat_img = 2048                             # features in each image
        model.eval()                                # method of evaluation (inference)

        return model, preprocess, feat_img

    elif cnn_name == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1       # database IMAGENET
        preprocess = weights.transforms()           # pre-processing transformations applied to input data
        model = vgg16(weights=weights)              # CNN VGG16
        model.classifier[6] = nn.Identity()         # remove last layer -> [out] (B, 4096)
        feat_img = 4096                             # features in each image
        model.eval()                                # method of evaluation (inference)

        return model, preprocess, feat_img

    else:
        raise ValueError(f"Unknown CNN name: {cnn_name}. Use 'resnet50' or 'vgg16'.")