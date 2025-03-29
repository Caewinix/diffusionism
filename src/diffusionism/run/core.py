from typing import Optional
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
import inyaml
from ..runner import running


torch.set_float32_matmul_precision('high')


def get_parser():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--train-config', '--train', type=str, default=None, required=False, help='The mode of this program.')
    parser.add_argument('--val-config', '--val', type=str, default=None, required=False, help='The mode of this program.')
    parser.add_argument('--test-config', '--test', type=str, default=None, required=False, help='The mode of this program.')
    parser.add_argument('-m', '--model-config', '--model', type=str, required=True, help='The model of this process.')
    parser.add_argument('-r', '--runner-config', '--runner', type=str, default=None, required=False, help='The model of this process.')
    parser.add_argument('-n', '--name', '--experiment-name', type=str, default='default', required=False, help='The model of this process.')
    parser.add_argument('-v', '--version', type=str, default='', required=False, help='The model of this process.')
    parser.add_argument('-c', '--checkpoints-root', type=str, default='./saved_checkpoints', required=False, help='The model of this process.')
    parser.add_argument('-l', '--logs-root', type=str, default='./logs', required=False, help='The model of this process.')
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, help='The seed of the random.')
    return parser


parser = get_parser()


def load_running_config(
    config: inyaml.namespace.Namespace,
    experiment_name: Optional[str] = None,
    checkpoints_root: Optional[str] = None,
    logs_root: Optional[str] = None,
    version: Optional[str] = None,
):
    settings = config.settings
    if settings is not None:
        del config.settings
        config.update(settings)
    if experiment_name is not None:
        config.experiment_name = experiment_name
    if checkpoints_root is not None:
        config.checkpoints_root = checkpoints_root
    if logs_root is not None:
        config.logs_root = logs_root
    if version is not None:
        config.version = version
    return config


def set_seed(seed):
    pl.seed_everything(seed, verbose=False, workers=True)


def main(args):
    set_seed(args.seed)
    with open(args.model_config, 'r') as file:
        model_config = inyaml.load(file)
    if args.runner_config is None:
        runner_config = model_config
    else:
        with open(args.runner_config, 'r') as file:
            runner_config = inyaml.load(file)
            runner_config.runner = runner_config.runner(**model_config)
    
    if args.test_config is not None:
        if args.train_config is not None or args.val_config is not None:
            raise RuntimeError("when '--test-config' is specified, '--train-config' or '--val-config' should not be provided")
        
        with open(args.test_config, 'r') as file:
            set_seed(torch.initial_seed())
            test_config = load_running_config(
                inyaml.load(file),
                experiment_name=args.name,
                logs_root=args.logs_root,
                version=args.version
            )
        test_config.update(runner_config)
        running.test(**test_config)
        
    else:
        if args.train_config is not None:
            with open(args.train_config, 'r') as file:
                train_config = load_running_config(
                    inyaml.load(file),
                    experiment_name=args.name,
                    checkpoints_root=args.checkpoints_root,
                    logs_root=args.logs_root,
                    version=args.version
                )
            train_config.update(runner_config)
            if args.val_config is not None:
                with open(args.val_config, 'r') as file:
                    set_seed(torch.initial_seed())
                    val_config = load_running_config(inyaml.load(file))
                    del val_config.individual
                train_config.update(val_config)
            running.train(**train_config)
        
        elif args.val_config is not None:
            with open(args.val_config, 'r') as file:
                set_seed(torch.initial_seed())
                val_config = load_running_config(
                    inyaml.load(file),
                    experiment_name=args.name,
                    checkpoints_root=args.checkpoints_root,
                    logs_root=args.logs_root,
                    version=args.version
                )
                individual = val_config.individual
                del val_config.individual
                val_config.update(individual)
            val_config.update(runner_config)
            running.val(**val_config)