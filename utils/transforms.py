from torchvision import transforms


def get_transforms(preprocess):
    """
    Return:
        * train_transform: with data augmentation (train)
        * eval_transform: no augmentation (val/test)
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.10, contrast=0.20), # test: (0,0), (0.10, 0.20) and (0.10, 0.30)
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = preprocess

    return train_transform, eval_transform