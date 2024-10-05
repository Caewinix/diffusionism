from typing import Union, Callable, Optional, List, Iterable, Mapping
import os
from copy import copy
import re
import bisect
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.optim.optimizer import Optimizer, params_t
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from .diffusion_runner import DiffusionRunner


def _get_name_and_version(name: str = 'default', version: Union[int, str, None] = ''):
    if version == '':
        return '', name
    return name, version


def _distinguish_step_and_epoch(step_interval, epoch_interval, step_interval_name = 'step_interval', epoch_interval_name = 'epoch_interval'):
    if step_interval is not None:
        if step_interval >= 0:
            every_n_train_steps = step_interval
            every_n_epochs = None
        else:
            every_n_train_steps = None
            every_n_epochs = -int(step_interval)
    elif epoch_interval is not None:
        if epoch_interval >= 0:
            every_n_train_steps = None
            every_n_epochs = epoch_interval
        else:
            raise ValueError(f"if '{epoch_interval_name}' is given, it should be greater than or equal to 0, but got {step_interval}")
    else:
        raise ValueError(f"one of '{step_interval_name}' and '{epoch_interval_name}' should be `None`")
    return every_n_train_steps, every_n_epochs


def _checkpoints_format_extract(name: str, format: str):
    epoch_re = format.replace('{epoch}', '(\d+)')
    match = re.match(epoch_re, name)
    if match is not None:
        epoch = match.groups(1)
    else:
        epoch = None
    step_re = format.replace('{step}', '(\d+)')
    match = re.match(step_re, name)
    if match is not None:
        step = match.groups(1)
    else:
        step = None
    if epoch is None:
        if step is None:
            return name
        return step
    elif step is None:
        return epoch
    return epoch, step, name


def _tensorboard_vis(
    summary_writer: SummaryWriter,
    images: Mapping[str, Iterable[torch.Tensor]],
    metrics: Mapping[str, Iterable[torch.Tensor]],
    cmaps: Mapping[str, str],
    num_rows: int = 1,
    figure_size = (25, 25)
):
    keys = list(images.keys())
    
    max_length = 0
    for values in images.items():
        length = len(values)
        if max_length < length:
            max_length = length
    
    keys_length = len(keys)
    num_columns = keys_length // num_rows
    
    plots = [plt.subplots(num_rows, num_columns, figsize=figure_size) for _ in range(max_length)]
    num_metrics = len(metrics)

    for i, key in enumerate(keys):
        for j, image in enumerate(images[key]):
            plots[j][1][i].set_title(key, fontsize=20)
            plots[j][1][i].imshow(image, cmap=cmaps[key], vmax=1, vmin=0)
            plots[j][1][i].axis('off')
            if i == keys_length - 1:
                out_str = ''
                for k, (key, value) in enumerate(metrics.items()):
                    out_str += f'{key}: {value[j]}'
                    if k != num_metrics - 1:
                        out_str += ', '
                plots[j][0].suptitle(out_str, fontsize=15)
                plots[j][0].tight_layout()
                summary_writer.add_figure('Comparison', plots[j][0], j)
    
    return summary_writer


def train(
    runner: DiffusionRunner,
    optimizer: Union[Optimizer, Callable[[params_t], Optimizer]],
    lr_scheduler: Union[LRScheduler, Callable[[Optimizer], LRScheduler]],
    train_data_loader: DataLoader,
    val_data_loader: Optional[DataLoader] = None,
    devices: Union[List[int], str, int] = "auto",
    limit_val_batches: Union[int, float, None] = None,
    monitor: Optional[str] = None,
    val_epoch_interval: Optional[int] = 1,
    val_step_interval: Optional[int] = None,
    saving_epoch_interval: Optional[int] = 1,
    saving_step_interval: Optional[int] = None,
    max_epochs: int = 7500,
    precision: str = 'bf16-mixed',
    grad_clip_norm: float = 1.,
    weight_path: Optional[str] = None,
    checkpoints_root: str = './saved_checkpoints',
    logs_root: str = './logs',
    experiment_name: str = 'default',
    version: Union[int, str, None] = ''
):
    """Trains the diffusion model.

    Args:
        runner (DiffusionRunner): The instance of :class:`DiffusionRunner`.
        optimizer (Union[Optimizer, Callable[[params_t], Optimizer]]): An optimizer or a function that
            accepts module parameters and returns an optimizer.
        lr_scheduler (Union[LRScheduler, Callable[[Optimizer], LRScheduler]]): The learning rate
            scheduler or a function that receives the optimizer and returns a learning rate scheduler.
        train_data_loader (DataLoader): The data loader for the training purpose.
        val_data_loader (Optional[DataLoader]): The data loader for the validation.
            If ``None``, no validation will be executed.
        devices: The devices to use. Can be set to a positive number (int or str),
            a sequence of device indices.
        limit_val_batches (Union[int, float, None]): How much of validation dataset
            to check (float = fraction, int = num_batches).
        monitor (Optional[str]): The metric that should be monitored for storage.
        val_epoch_interval (Optional[int]): The validation interval at the epoch level,
            indicating the validation execution after how many epochs, which should be
            always greater than or equal to zero. If ``None``, meaning it may be at
            step level, or just no validation.
        val_step_interval (Optional[int]): The validation interval at the step level,
            indicating the validation execution after how many steps, it will validate
            every -:param:`val_step_interval` if :param:`val_step_interval` is less
            than ``0``. If ``None``, meaning it may be at epoch level, or just no validation.
        saving_epoch_interval (Optional[int]): The saving interval at the epoch level,
            indicating the storage after how many epochs, which should be  always
            greater than or equal to zero. If ``None``, meaning it may be at
            step level, or just no saving.
        saving_step_interval (Optional[int]): The saving interval at the step level,
            indicating the storage after how many steps, it will save every
            -:param:`val_step_interval` if :param:`val_step_interval` is less
            than ``0``. If ``None``, meaning it may be at epoch level, or just no saving.
        max_epochs (int): The maximum epochs the runner will train.
        precision (str): The precision label about this execution.
        grad_clip_norm (float): The norm grad clip value.
        weight_path (Optional[str]): The path of the pretrained weight. If ``None``, the model
            will not load any pretrained weights.
        checkpoints_root (str): The root that the checkpoints will be saved.
        logs_root (str): The root that the log files will be located in.
        experiment_name (str): The name of this experiment.
        version (Union[int, str, None]): The version of this experiment, which can be
            ``int`` or ``str``, ``""`` (empty string) means providing no version.
            If ``None``, the runner will infer the next version of this experiment from
            the experiment directory.
    
    """
    experiment_name, version = _get_name_and_version(experiment_name, version)
    
    runner.set_optimizer(optimizer)
    runner.set_lr_scheduler(lr_scheduler)
    if weight_path is not None:
        runner.load_from_checkpoint(checkpoint_path=weight_path)
    
    dirpath = os.path.join(checkpoints_root, experiment_name, version)
    
    callbacks = []
    if val_data_loader is not None and (val_step_interval is not None or val_epoch_interval is not None):
        every_n_train_steps, every_n_epochs = _distinguish_step_and_epoch(
            val_step_interval,
            val_epoch_interval,
            'val_step_interval',
            'val_epoch_interval'
        )
        if every_n_epochs is not None:
            val_check_interval = every_n_epochs * len(train_data_loader)
        else:
            val_check_interval = every_n_train_steps
        val_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename='best',
            save_last=saving_step_interval is None and saving_epoch_interval is None,
            enable_version_counter=False,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            save_top_k=1,
            monitor=monitor,
            mode='max',
        )
        callbacks.append(val_checkpoint_callback)
    else:
        val_check_interval = 0
    if saving_step_interval is not None or saving_epoch_interval is not None:
        every_n_train_steps, every_n_epochs = _distinguish_step_and_epoch(
            saving_step_interval,
            saving_epoch_interval,
            'saving_step_interval',
            'saving_epoch_interval'
        )
        saving_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename='{epoch}-{step}',
            save_last=True,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            save_top_k=-1,
            monitor=monitor,
            mode='max',
            save_on_train_epoch_end=True
        )
        callbacks.append(saving_checkpoint_callback)
    
    tensor_board_logger = pl_loggers.TensorBoardLogger(
        save_dir=logs_root,
        version=version,
        name=experiment_name
    )
    
    trainer = pl.Trainer(
        devices=devices,
        callbacks=callbacks,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=grad_clip_norm,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        logger=tensor_board_logger,
        deterministic=True
    )
    trainer.fit(
        runner,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
        ckpt_path='last'
    )


def val(
    runner: DiffusionRunner,
    val_data_loader: DataLoader,
    devices: Union[List[int], str, int] = 1,
    start_epoch: Optional[int] = 0,
    start_step: Optional[int] = None,
    precision: str = 'bf16-mixed',
    checkpoints_root: str = './saved_checkpoints',
    filename_format: str = '{epoch}-{step}',
    logs_root: str = './logs',
    experiment_name: str = 'default',
    version: Union[int, str, None] = ''
):
    """Validates the diffusion model.

    Args:
        runner (DiffusionRunner): The instance of :class:`DiffusionRunner`.
        val_data_loader (DataLoader): The data loader for the validation.
        devices: The devices to use. Can be set to a positive number (int or str),
            a sequence of device indices.
        start_epoch (Optional[int]): The start epoch of the checkpoints list. If ``None``,
            the start index will rely on the :param:`start_step` or start from ``0``.
        start_step (Optional[int]): The start step of the checkpoints list. If ``None``,
            the start index will rely on the :param:`start_epoch` or start from ``0``.
        precision (str): The precision label about this execution.
        checkpoints_root (str): The root that the checkpoints will be saved.
        filename_format (str): The checkpoint filename format, keyword `{epoch}` and `{step}`
            can be used to determine the designated counterpart.
        logs_root (str): The root that the log files will be located in.
        experiment_name (str): The name of this experiment.
        version (Union[int, str, None]): The version of this experiment, which can be
            ``int`` or ``str``, ``""`` (empty string) means providing no version.
            If ``None``, the runner will infer the next version of this experiment from
            the experiment directory.
    
    """
    experiment_name, version = _get_name_and_version(experiment_name, version)
    
    checkpoints_dir = os.path.join(checkpoints_root, experiment_name, version)
    checkpoint_filenames = sorted(os.listdir(checkpoints_dir), key=_checkpoints_format_extract)
    def get_index(key):
        return bisect.bisect_left(checkpoint_filenames, key, key=lambda x: _checkpoints_format_extract(x, filename_format))
    if start_epoch is None:
        if start_step is None:
            index = 0
        else:
            index = get_index(start_step)
    elif start_step is None:
        index = get_index(start_epoch)
    else:
        index = get_index((start_epoch, start_step, ''))
    checkpoint_filenames = checkpoint_filenames[index:]
    
    tensor_board_logger = pl_loggers.TensorBoardLogger(
        save_dir=logs_root,
        version=version,
        name=experiment_name,
        sub_dir='val'
    )
    trainer = pl.Trainer(
        devices=devices,
        precision=precision,
        val_check_interval=0,
        logger=tensor_board_logger,
        deterministic=True
    )
    best_metrics = None
    best_metrics_filename = {}
    best_metrics_all = {}
    with tqdm(checkpoint_filenames, dynamic_ncols=True) as tqdm_ckpt_filenames:
        for ckpt_filename in tqdm_ckpt_filenames:
            metrics = trainer.validate(
                runner,
                val_data_loader,
                os.path.join(checkpoints_dir, ckpt_filename)
            )
            metrics = runner.evaluation_epoch_end(metrics, False)
            if best_metrics is None:
                best_metrics = copy(metrics)
            num_metrics = len(metrics)
            out_str = f'{ckpt_filename} - '
            for i, (key, value) in enumerate(metrics.items()):
                out_str += f'{key}: {value}'
                if i != num_metrics - 1:
                    out_str += ', '
                if best_metrics[key] < value:
                    best_metrics[key] = value
                    best_metrics_filename[key] = ckpt_filename
                    best_metrics_all[key] = copy(metrics)
                    best_metrics_all[key].pop(key)
            tqdm.write(out_str)
    tqdm.write("BEST RESULTS:")
    for best_key, filename in best_metrics_filename.items():
        tqdm.write(f'~ Best {best_key} ({filename}): {best_metrics[best_key]}')
        out_str = '\tOthers - '
        for i, (key, value) in enumerate(best_metrics_all[best_key].items()):
            out_str += f'{key}: {value}'
            if i != num_metrics - 1:
                out_str += ', '
        tqdm.write(out_str)


def test(
    runner: DiffusionRunner,
    test_data_loader: DataLoader,
    checkpoint_path: str,
    devices: Union[List[int], str, int] = 1,
    precision: str = 'bf16-mixed',
    output_root: Optional[str] = None,
    logs_root: str = './logs',
    experiment_name: str = 'default',
    version: Union[int, str, None] = ''
):
    """Tests the diffusion model.

    Args:
        runner (DiffusionRunner): The instance of :class:`DiffusionRunner`.
        test_data_loader (DataLoader): The data loader for test.
        checkpoint_path (str): The checkpoint path that is expected to be tested.
        devices: The devices to use. Can be set to a positive number (int or str),
            a sequence of device indices.
        precision (str): The precision label about this execution.
        output_root (Optional[str]): The root of the output generations and original
            targets. If ``None``, they will not be output.
        logs_root (str): The root that the log files will be located in.
        experiment_name (str): The name of this experiment.
        version (Union[int, str, None]): The version of this experiment, which can be
            ``int`` or ``str``, ``""`` (empty string) means providing no version.
            If ``None``, the runner will infer the next version of this experiment from
            the experiment directory.
    
    """
    experiment_name, version = _get_name_and_version(experiment_name, version)
    
    tensor_board_logger = pl_loggers.TensorBoardLogger(
        save_dir=logs_root,
        version=version,
        name=experiment_name,
        sub_dir='test'
    )
    trainer = pl.Trainer(
        devices=devices,
        precision=precision,
        val_check_interval=0,
        logger=tensor_board_logger,
        deterministic=True
    )

    runner.use_test_collection()
    metrics = trainer.test(
        runner,
        test_data_loader,
        checkpoint_path
    )
    metrics = runner.evaluation_epoch_end(metrics, False)
    generation, target = runner.get_test_collection()
    
    comparison = {
        'generation' : generation.permute(0, 2, 3, 1).cpu(),
        'target' : target.permute(0, 2, 3, 1).cpu(),
        'error': torch.abs(generation - target).permute(0, 2, 3, 1).cpu()
    }
    
    if output_root is not None:
        output_dir = os.path.join(output_root, experiment_name, version)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(comparison, os.path.join(output_dir, 'test_collection.pt'))
    
    _tensorboard_vis(
        tensor_board_logger.experiment,
        comparison,
        metrics,
        {'generation' : 'gray', 'target' : 'gray', 'error': 'Reds'}
    )
    
    num_metrics = len(metrics)
    out_str = ''
    for i, (key, value) in enumerate(metrics.items()):
        out_str += f'{key}: {value}'
        if i != num_metrics - 1:
            out_str += ', '
    tqdm.write(out_str)