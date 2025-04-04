import os
from typing import List
import shutil
from collections import defaultdict
import argparse
import copy
from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from src.utils import instantiate_from_config, get_timestamp, setup_logger


logger = setup_logger(__file__)


def instantiate_callbacks(callback_configs: ListConfig):
    callbacks = []
    for callback_cfg in callback_configs:
        callbacks.append(instantiate_from_config(callback_cfg))
    
    return callbacks


def get_dataloaders(config, args):
    train_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'train'})
    val_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'val'})
    test_ds = instantiate_from_config(config.dataset, extra_kwargs={'split': 'test'})

    dataloader_config = copy.copy(config.dataloader)
    val_batch_size = dataloader_config.pop('val_batch_size', dataloader_config.batch_size)
    train_dataloader = DataLoader(train_ds, **dataloader_config, shuffle= not args.no_shuffle_train, drop_last=True)
    dataloader_config.batch_size = val_batch_size
    val_dataloader = DataLoader(val_ds, **dataloader_config, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_ds, **dataloader_config, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


def _preprocess_config(config, args, unknown_args):
    # global logger
    def set_config_key_value(inplace_dict, key_path, value):
        def bfs_set_config_key_value(inplace_dict, key, value):
            at_least_one_kv_is_set = False
            if not isinstance(inplace_dict, (DictConfig, dict)):
                return False
            if key in inplace_dict.keys():
                inplace_dict[key] = value
                at_least_one_kv_is_set = True
            for v in inplace_dict.values():
                if isinstance(v, (DictConfig, dict)):
                    at_least_one_kv_is_set |= bfs_set_config_key_value(inplace_dict=v, key=key, value=value)
                elif isinstance(v, ListConfig):
                    for item in v:
                        at_least_one_kv_is_set |= bfs_set_config_key_value(inplace_dict=item, key=key, value=value)
            return at_least_one_kv_is_set

        keys = key_path.split('.')  # e.g., dataset.a.b=1
        len_keys = len(keys)
        if len_keys == 1:  # e.g., batch_size=32
            success = bfs_set_config_key_value(inplace_dict, key=key_path, value=value)
            if success:
                return 
            else:
                raise ValueError(f'{key_path} is not found in config')

        # else len_keys > 1:
        for key_idx in range(len_keys - 1):
            inplace_dict = inplace_dict[keys[key_idx]]

            if isinstance(inplace_dict, ListConfig):
                for item in inplace_dict:
                    for sub_key_idx in range(key_idx + 1, len_keys - 1):
                        item = item[keys[sub_key_idx]]
                    item[keys[-1]] = value
                return

        inplace_dict[keys[-1]] = value

    # set unknown args to config
    for unknown in unknown_args:
        k, v = unknown.split('=')
        try:
            v = int(v)  # maybe int has the highest priority
        except:
            try:
                v = float(v)
            except:
                # Python constants: True, False, None
                # it should not be v = bool(v) as bool('False') -> True
                if (vlower := v.lower()) == 'true':
                    v = True
                elif vlower == 'false':
                    v = False
                elif vlower == 'none':
                    v = None
                # else v = v, the str itself
        set_config_key_value(config, k, v)

    # devices
    devices = args.devices
    if devices is None:
        config.trainer.accelerator = 'cpu'  # bet you won't run into this line
    else:
        config.trainer.devices = [int(rank) for rank in devices.split(',')]

    # set project name and signature for logging
    if args.no_log:
        config.trainer.logger = False
    else:
        config.trainer.logger.save_dir = f'logs/{args.model}'
        config.trainer.logger.name = f'{args.dataset}'
        config.trainer.logger.version = get_timestamp() + (f'_{args.log_suffix}' if args.log_suffix != '' else '')

    # batch size for ddp
    total_bs = config.dataloader.batch_size
    num_devices = len(config.trainer.devices)
    bs_per_device = total_bs // num_devices
    real_bs = bs_per_device * num_devices
    if real_bs != total_bs:
        logger.warning(f'real batch size is {real_bs}')
    config.dataloader.batch_size = bs_per_device

    # epoch scaling: scaling up the epoch length while reducing the number of epochs
    # this is useful when an epoch is too short and val is too frequent
    epoch_scaling = config.dataset.get('epoch_scaling')
    if epoch_scaling is not None and epoch_scaling != 1:
        config.trainer.max_epochs = int(config.trainer.max_epochs / epoch_scaling)
        logger.info(f'Training epoch length is scaled by {epoch_scaling}, thus the num of epochs is decreased to {config.trainer.max_epochs}')
    
    # process the config here
    config = preprocess_config_hook(config)

    logger.info(f'running with config: {config}')
    return config


def preprocess_config_hook(config):
    return config


def get_processed_args_and_config():
    args, unknown_args = get_args()

    OmegaConf.register_new_resolver("eval", eval)

    # load trainer config
    trainer_config = OmegaConf.load(f'src/configs/trainer/{args.trainer}.yaml')

    # load model config
    model_config = OmegaConf.load(f'src/configs/models/{args.model}.yaml')
    config = OmegaConf.merge(trainer_config, model_config)

    # load dataset config
    dataset_config = OmegaConf.load(f'src/configs/datasets/{args.dataset}.yaml')
    config = OmegaConf.merge(config, DictConfig(dataset_config))
    OmegaConf.resolve(config)

    config = _preprocess_config(config, args, unknown_args)
    
    return args, config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        default='motion_clip'
    )

    parser.add_argument(
        '--dataset',
        default='motion_clip'
    )

    parser.add_argument(
        '--trainer',
        default='default'  # actually this is the only trainer
    )

    parser.add_argument(
        '--devices',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--resume_ckpt_path',
        type=str,
        default=None
    )

    parser.add_argument(
        '--load_ckpt_path',
        type=str,
        default=None
    )

    parser.add_argument(
        '--no_log',  # when debugging, setting this to False helps. (Recommend to add this to launch.json)
        help='disable training log',
        action='store_true'
    )

    parser.add_argument(
        '--log_suffix',
        help='append a suffix to log dir',
        default=''
    )

    parser.add_argument(
        '--no_shuffle_train',
        action='store_true'
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, config = get_processed_args_and_config()
    pl.seed_everything(config.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config, args)
    epoch_length = len(train_dataloader) // len(config.trainer.devices)
    config.model.training_kwargs['num_training_steps'] = epoch_length * config.trainer.max_epochs

    model: pl.LightningModule = instantiate_from_config(config.model, extra_kwargs={"all_config": config})
    if p := args.load_ckpt_path:
        model.load_state_dict(state_dict=torch.load(p, map_location='cpu')['state_dict'], strict=False)

    trainer: pl.Trainer = instantiate_from_config(config.trainer, extra_kwargs={'callbacks': instantiate_callbacks(config.callbacks)})

    try:
        try:
            if trainer.global_rank == 0:
                shutil.copytree('src', os.path.join(trainer.logger.log_dir, 'src_backup'))  # backup src directory
        except: pass

        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.resume_ckpt_path)

        # evaluation
        results = trainer.test(ckpt_path='best', dataloaders=test_dataloader)[0]  # the first dataloader
        logger = setup_logger('results', log_file=f'{trainer.logger.log_dir}/eval_after_train.log')
        logger.info(f'evaluation results: {results}')

    except Exception as e:
        raise e
    else:
        # mark log dir as trained
        if trainer.global_rank == 0:
            shutil.move(trainer.logger.log_dir, trainer.logger.log_dir + '_trained')


if __name__ == '__main__':
    main()
