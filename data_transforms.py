from typing import List, Tuple

from torchvision import transforms


# Statistiques ImageNet, partagees par l'entrainement et l'evaluation.
NORM_MEAN: List[float] = [0.485, 0.456, 0.406]
NORM_STD: List[float] = [0.229, 0.224, 0.225]


def build_train_transform(img_size: Tuple[int, int]) -> transforms.Compose:
    """Augmentation aleatoire appliquee a chaque epoch."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def build_eval_transform(img_size: Tuple[int, int]) -> transforms.Compose:
    """Pretraitement deterministe pour la validation et l'inference.

    Doit rester aligne sur la normalisation de `build_train_transform`, sinon
    le modele voit en eval une distribution differente de celle apprise.
    """
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
