"""
Obtained from https://github.com/nis-research/afa-augment/blob/main/utils.py

"""

import cv2
from torchvision import transforms as T
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import wandb
import numpy as np

################################# Added for DL #########################################

class ColorspaceTransform(T.Compose):

    def __init__(self, colorspace):
        self.colorspace = colorspace

    def __call__(self, img):
        # Convert PIL image to NumPy array
        array = np.array(img)

        # Convert the color space using OpenCV
        transformed_array = cv2.cvtColor(array, self.colorspace)

        # Convert the transformed NumPy array back to PIL image
        transformed_img = Image.fromarray(transformed_array)

        return transformed_img

#################################  Until here  #########################################

class MyWandBLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        if self._log_model:
            if checkpoint_callback.best_model_path:
                # get the best model path
                best_model_path = checkpoint_callback.best_model_path
                # log the best model
                wandb.save(best_model_path, base_path=wandb.run.dir)


def get_standard_transforms(dataset, img_sz, premix='none', colorspace=None):
    if dataset in ['c10', 'c100']:
        # train_transform = [T.RandomCrop(img_sz, padding=4), T.RandomHorizontalFlip(),
        #                    T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10), T.ToTensor()]

        train_transform = [
            # T.RandomCrop(img_sz, padding=4),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        # if premix == 'ta':
        #     print('We are using TrivialAugmentWide()!')
        #     train_transform.append(T.TrivialAugmentWide())

        train_transform = T.Compose(train_transform)

        test_transform = T.Compose([
            T.ToTensor(),
        ])

    ################################# Added for DL #########################################
    elif dataset in ['c10-HSV', 'c10-HLS', 'c10-YUV', 'c10-LUV']:
        # colorspace_transform = T.transforms.Lambda(
        #     lambda img: Image.fromarray(cv2.cvtColor(np.array(img), colorspace)))
        train_transform = [ColorspaceTransform(colorspace),
                           T.ToTensor()] #T.RandomCrop(img_sz, padding=4), T.RandomHorizontalFlip(),
        test_transform = [ColorspaceTransform(colorspace), T.ToTensor()]


        train_transform = T.Compose(train_transform)
        test_transform = T.Compose(test_transform)
    #################################  Until here  #########################################

    elif dataset in ['in', 'in100']:
        train_transform = [
            T.RandomResizedCrop(img_sz, antialias=True), T.RandomHorizontalFlip(),
        ]

        if premix == 'ta':
            train_transform.append(T.TrivialAugmentWide())

        train_transform.append(T.ToTensor())

        train_transform = T.Compose(train_transform)

        test_transform = T.Compose([
            T.Resize(256, antialias=True), T.CenterCrop(img_sz),
            T.ToTensor(),
        ])
    elif dataset == 'GPS':
        train_transform = [T.ToTensor()]
        test_transform = [T.ToTensor()]
    else:
        raise NotImplementedError
    return test_transform, train_transform