import albumentations as A
from albumentations.pytorch import ToTensorV2

UNIFIED_SIZE = (128, 128)
COMMON_MEAN = 0.0
COMMON_STD = 1.0

train_transform = A.Compose(
    [
        A.Resize(*UNIFIED_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
        A.Normalize(mean=COMMON_MEAN, std=COMMON_STD),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(*UNIFIED_SIZE),
        A.Normalize(mean=COMMON_MEAN, std=COMMON_STD),
        ToTensorV2(),
    ]
)

brats_transform = train_transform
lidc_transform = train_transform

brats_transform_light = A.Compose(
    [
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ]
)
