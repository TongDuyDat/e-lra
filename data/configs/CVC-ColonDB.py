# config.py
import albumentations as A

DATASET_SOURCE = "data/data_benmarks/CVC-ColonDB"
IMAGE_SIZE = (256, 256)
NORMALIZE = True

TRANSFORM_PIPELINE = {
    "train": A.Compose(
        [
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1.0, sigma=50, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=0.2, mode="camera", p=1.0),
                ],
                p=0.3,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        ],
        is_check_shapes=False,
    ),
    "val": A.Compose(
        [A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])],
        is_check_shapes=False,
    ),
}
