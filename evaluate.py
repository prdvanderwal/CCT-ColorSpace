#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 08/02/2024 16:52:56

@author: prdvanderwal
"""

# imports
import os
import wandb
import cv2
import argparse

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
import torchvision.transforms as T


from utils import MyWandBLogger, get_standard_transforms
from src.cct_lightning import *
from dsets import get_dataset, get_c_dataset
from models.cct import *


def main(config, weights=None):
    ################################# Added for DL #########################################
    if torch.cuda.is_available():
        print("GPU available")


    # Reproducability settings
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(config.seed, workers=True)

    # Login at wandb
    wandb.login(key=config.api_key)


    checkpoint_path = config.path

    logger = MyWandBLogger(project=config.project, name=config.name)
    logger.experiment


    # create learning rate logger
    lr_logger = LearningRateMonitor(logging_interval="epoch")

    # # Init trainer
    trainer = pl.Trainer(
        # logger=logger,
        # callbacks=[lr_logger, model_checkpoint, ModelSummary(max_depth=-1)],
        log_every_n_steps=50,
        accelerator=config.accelerator,
        devices=1,
        deterministic=True,
        enable_progress_bar=True,
        max_epochs=config.epochs,
    )

    ################################# Until here #########################################

    # Get all image data
    dataset_class = get_dataset(config.dataset)
    img_sz = dataset_class.image_size
    num_classes = dataset_class.num_classes
    normalise_transform = T.Normalize(mean=dataset_class.mean, std=dataset_class.std)

    # Init model_class
    model_class = CCT_7(extra_LN=config.extra_LN)

    # Data transforms
    test_transform, train_transform = get_standard_transforms(config.dataset, img_sz,
                                                              premix=config.enable_aug.premix,
                                                              colorspace=config.colorconversion)



    # Create dataset and loaders
    train_dataset, validation_dataset = [
        dataset_class(root=config.data_dir, train=train, transform=transform)
        for train, transform in [
            (True, train_transform), (False, test_transform) #None if using_wrapper else
        ]
    ]

    train_loader, validation_loader = [
        DataLoader(
            dset, batch_size=config.batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            persistent_workers=True
        )
        for dset, num_workers, shuffle in [(train_dataset, 6, True), (validation_dataset, 4, False)]
    ]

    ################################# Added for DL #########################################
    # Initialize your model
    model = TrainingModule.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, model=model_class,
                                                num_classes=num_classes,
                                                normalisation=normalise_transform)

    # Set the model to evaluation mode
    model.eval()

    # trainer.test(model=model, dataloaders=validation_loader)
    test_accs = {}

    classwise_table = wandb.Table(columns=["Class", "accuracy"])

    clean_test_log = model.manual_test(validation_loader, name='clean')

    for value,key in clean_test_log['classwise_acc'].items():
        classwise_table.add_data(key, value)

    wandb.log({'classwise_acc' : classwise_table})
    ################################# Until here #########################################


    #
    my_table = wandb.Table(columns=["corruption", "severity", "accuracy"])
    second_table = wandb.Table(columns=["corruption", "accuracy"])

    my_table.add_data("clean", 0, clean_test_log['val_acc'])
    test_accs['clean'] = {
        0: clean_test_log['val_acc']
    }

    severities = (1, 2, 3, 4, 5)

    c_dataset_class = get_c_dataset(config.dataset)
    for corruption in c_dataset_class.corruptions:
        for severity in severities:
            c_s_dst = c_dataset_class(
                config.data_dir, transform=test_transform, severity=severity, corruption=corruption
            )

            c_s_loader = DataLoader(
                c_s_dst,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            c_s_log = model.manual_test(c_s_loader, name=f'{corruption}_{severity}'.ljust(25))

            my_table.add_data(corruption, severity, c_s_log['val_acc'])
            test_accs[corruption] = test_accs.get(corruption, {})
            test_accs[corruption][severity] = c_s_log['val_acc']


    wandb.log({'corruption_table': my_table})

    ################################# Added for DL #########################################
    second_table.add_data('clean', clean_test_log['val_acc'])
    for corruption in test_accs.keys():
        if corruption != 'clean:':
            corr_acc = sum([acc for acc in test_accs[corruption].values()]) / len(test_accs[corruption])
            second_table.add_data(corruption, corr_acc)

    wandb.log({'Corruption_acc_table': second_table})
    ################################# Until here   #########################################

    avg_test_acc = sum(
        [
            sum([acc for acc in test_accs[corruption].values()]) / len(test_accs[corruption])
            for corruption in test_accs if corruption != 'clean'
        ]
    ) / (len(test_accs) - 1)
    wandb.log({'corr_acc': avg_test_acc})

    wandb.finish()


def cc_test(model, config, test_transform, val_loader, severities=(4,)):
    test_accs = {}

    # Customized
    ckpt_path = config.checkpoint

    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    clean_test_log = model.manual_test(val_loader, name='clean'.ljust(25))
    my_table = wandb.Table(columns=["corruption", "severity", "accuracy"])

    my_table.add_data("clean", 0, clean_test_log['val_acc'])
    test_accs['clean'] = {
        0: clean_test_log['val_acc']
    }

    c_dataset_class = get_c_dataset(config.dataset)
    for corruption in c_dataset_class.corruptions:
        for severity in severities:
            c_s_dst = c_dataset_class(
                config.data_dir, transform=test_transform, severity=severity, corruption=corruption
            )

            c_s_loader = DataLoader(
                c_s_dst,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )

            c_s_log = model.manual_test(c_s_loader, name=f'{corruption}_{severity}'.ljust(25))

            my_table.add_data(corruption, severity, c_s_log['val_acc'])
            test_accs[corruption] = test_accs.get(corruption, {})
            test_accs[corruption][severity] = c_s_log['val_acc']

    wandb.log({'corruption_table': my_table})

    return test_accs


if __name__ == '__main__':
    ################################# Added for DL #########################################
    parser = argparse.ArgumentParser(description='Color ViT')
    parser.add_argument('--colorspace', type=str, default=None, choices=[None, 'HSV', 'HLS', 'LUV', 'YUV'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--extra_LN', action='store_true')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)

    # Parse the command-line arguments
    args = parser.parse_args()

    color_conversions = {
        None: {'conversion': None, 'reverse_conversion': None},
        'HSV': {'conversion': cv2.COLOR_RGB2HSV, 'reverse_conversion': cv2.COLOR_HSV2RGB},
        'HLS': {'conversion': cv2.COLOR_RGB2HLS, 'reverse_conversion': cv2.COLOR_HLS2RGB},
        'LUV': {'conversion': cv2.COLOR_RGB2LUV, 'reverse_conversion': cv2.COLOR_LUV2RGB},
        'YUV': {'conversion': cv2.COLOR_RGB2YUV, 'reverse_conversion': cv2.COLOR_YUV2RGB},
    }
    ################################# Until here   #########################################

    from pprint import pprint
    from config_utils import ConfigBuilder

    _config_params = [
        {
            'ds': 'c10', 'm': 'CCT7', 'use_jsd': False,
            'use_prime': False, 'use_augmix': False, 'in_mix': False, 'use_mix': False,
            'use_fourier': False, 'use_apr': False, 'attack': None, 'min_str': 0., 'mean_str': 10.,
            'colorspace': args.colorspace,
        },
    ]

    for i, _config_param in enumerate(_config_params):
        _config = ConfigBuilder.build(**_config_param)
        _config.num_workers = 6
        _config.project = "ViT-Color"
        _config.api_key = "27656ca0b0297d3f09e317d7a47bd97275cc33a1"
        _config.seed = args.seed
        _config.accelerator = "gpu"
        _config.num_classes = 10
        ################################# Added for DL #########################################
        _config.colorconversion = color_conversions[args.colorspace]['conversion']
        _config.reverse_colorconversion = color_conversions[args.colorspace]['reverse_conversion']
        _config.extra_LN = args.extra_LN

        if args.name:
            _config.name = args.name
        else:
            _config.name = f'CCT7_s{args.seed}_c{args.colorspace}'

        if args.run_id:
            _config.run_id = args.run_id

        if args.path:
            _config.path = args.path

        ################################# Until here   #########################################

        _weights = None

        pprint(_config.to_dict())
        main(_config, _weights)
