from typing import Union, Callable, Optional, List, Iterable, Mapping, Sequence, Any, Tuple
import os
from copy import copy
import re
import bisect
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint as IndistinguishableModelCheckpoint
from ...utils.workspace import Workspace
from ..model_runner import ModelRunner, params_t


class ModelCheckpoint(IndistinguishableModelCheckpoint):
    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
            id=id(self)
        )


class TensorBoardLogger(pl_loggers.TensorBoardLogger):
    @property
    def log_dir(self) -> str:
        """The directory for this run's tensorboard checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        if version == '':
            log_dir = os.path.join(self.root_dir, self.name)
        else:
            log_dir = os.path.join(self.root_dir, self.name, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir


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


def _get_match_format(m, sequence):
    value = m.group(1)
    sequence.append(value)
    return f"{value}=(?P<{value}>\d+)"


def _checkpoints_format_extract(name: str, format: str):
    tags = []
    pattern = re.sub(r'\{(\w+)\}', lambda m: _get_match_format(m, tags), format)
    if len(tags) == 0:
        return name
    tags.sort()
    match = re.match(pattern, name)
    if match is None:
        return (-1,)
    else:
        extraction = tuple(int(match.group(tag)) for tag in tags)
        return extraction


# def _tensorboard_vis(
#     summary_writer: SummaryWriter,
#     images: Mapping[str, Iterable[torch.Tensor]],
#     metrics: Mapping[str, Iterable[torch.Tensor]],
#     cmaps: Mapping[str, str],
#     num_rows: int = 1,
#     figure_size = (100, 100)
# ):
#     keys = list(images.keys())
    
#     max_length = 0
#     for value in images.values():
#         length = len(value)
#         if max_length < length:
#             max_length = length
    
#     keys_length = len(keys)
#     num_columns = keys_length // num_rows
    
#     # plots = [plt.subplots(num_rows, num_columns, figsize=figure_size) for _ in range(max_length)]
#     num_metrics = len(metrics)

#     for i in tqdm(range(max_length)):
#         figure, plot_arr = plt.subplots(num_rows, num_columns, figsize=figure_size)
#         for j, key in enumerate(keys):
#             image = images[key]
#             if len(image) > i:
#                 image = image[i]
#             else:
#                 continue
#             plot_arr[j].set_title(key, fontsize=50)
#             plot_arr[j].imshow(image.float(), cmap=cmaps[key], vmax=1, vmin=0)
#             plot_arr[j].axis('off')
#             if j == keys_length - 1:
#                 out_str = ''
#                 for k, (metric_key, value) in enumerate(metrics.items()):
#                     if len(value) > i:
#                         out_str += f'{metric_key}: {value[i]}'
#                         if k != num_metrics - 1:
#                             out_str += ', '
#                 figure.suptitle(out_str, fontsize=50)
#                 figure.tight_layout()
#                 summary_writer.add_figure('Comparison', figure, i)
#         plt.close(figure)
    
#     return summary_writer


def train(
    runner: ModelRunner,
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
    version: Union[int, str, None] = '',
    preparation: Optional[Callable[..., Any]] = None,
    finishing: Optional[Callable[..., Any]] = None
):
    """Trains the diffusion model.

    Args:
        runner (ModelRunner): The instance of :class:`ModelRunner`.
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
        preparation (Optional[Callable[..., Any]]): The function that will be called before
            any instructions.
        finishing (Optional[Callable[..., Any]]): The function that will be called after
            any instructions.

    """
    runner.set_optimizer(optimizer)
    runner.set_lr_scheduler(lr_scheduler)
    if weight_path is not None:
        runner.load_from_checkpoint(checkpoint_path=weight_path)
    
    dirpath = os.path.join(checkpoints_root, experiment_name, version)
    
    callbacks = []
    if monitor is None:
        monitor_key = None
        mode = 'min'
    else:
        if isinstance(monitor, str):
            mode = 'max'
        elif isinstance(monitor, Sequence) and len(monitor) == 2:
            monitor, mode = monitor
        elif isinstance(monitor, dict) and len(monitor) == 1:
            mode = next(monitor.values())
            monitor = next(monitor.keys())
        monitor_key = f'Val-{monitor}'
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
            monitor=monitor_key,
            mode=mode
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
            enable_version_counter=False,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            save_top_k=-1,
            monitor=monitor_key if val_data_loader is not None else None,
            mode=mode,
            save_on_train_epoch_end=True
        )
        callbacks.append(saving_checkpoint_callback)
    
    tensor_board_logger = TensorBoardLogger(
        save_dir=logs_root,
        version=version,
        name=experiment_name
    )
    
    if limit_val_batches is None and val_data_loader is not None:
        limit_val_batches = len(val_data_loader.dataset)
    
    has_trained = os.path.exists(os.path.join(dirpath, 'last.ckpt'))
    
    workspace = Workspace(
        devices=devices,
        callbacks=callbacks,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=grad_clip_norm,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        logger=tensor_board_logger,
        deterministic=True,
        num_sanity_val_steps=None if has_trained else 0
    )
    workspace.initialize_workspace(globals(), locals())
    workspace.set_workspace(preparation, finishing)
    workspace.fit(
        runner,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
        ckpt_path='last' if has_trained else None
    )


def val(
    runner: ModelRunner,
    val_data_loader: DataLoader,
    devices: Union[List[int], str, int] = 1,
    start_epoch: Optional[int] = 0,
    start_step: Optional[int] = None,
    precision: str = 'bf16-mixed',
    checkpoints_root: str = './saved_checkpoints',
    filename_format: str = '{epoch}-{step}',
    output_root: Optional[str] = None,
    need_images: bool = False,
    logs_root: str = './logs',
    experiment_name: str = 'default',
    version: Union[int, str, None] = '',
    preparation: Optional[Callable[..., Any]] = None,
    finishing: Optional[Callable[..., Any]] = None
):
    """Validates the diffusion model.

    Args:
        runner (ModelRunner): The instance of :class:`ModelRunner`.
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
        output_root (Optional[str]): The root of the output generations and original
            targets. If ``None``, they will not be output.
        need_images (bool) : The flag that controls whether to use the validation images.
        logs_root (str): The root that the log files will be located in.
        experiment_name (str): The name of this experiment.
        version (Union[int, str, None]): The version of this experiment, which can be
            ``int`` or ``str``, ``""`` (empty string) means providing no version.
            If ``None``, the runner will infer the next version of this experiment from
            the experiment directory.
        preparation (Optional[Callable[..., Any]]): The function that will be called before
            any instructions.
        finishing (Optional[Callable[..., Any]]): The function that will be called after
            any instructions.
    
    """
    checkpoints_dir = os.path.join(checkpoints_root, experiment_name, version)
    checkpoint_filenames = sorted(os.listdir(checkpoints_dir), key=lambda x: _checkpoints_format_extract(x, filename_format))
    def get_index(key):
        return bisect.bisect_left(checkpoint_filenames, key, key=lambda x: _checkpoints_format_extract(x, filename_format))
    if start_epoch is None:
        if start_step is None:
            index = 0
        elif '{epoch}' in filename_format:
            index = get_index((0, start_step))
        else:
            index = get_index((start_step,))
    elif start_step is None:
        index = get_index((start_epoch,))
    else:
        index = get_index((start_epoch, start_step))
    tqdm.write(f'Start validating from checkpoint "{checkpoint_filenames[index]}"')
    checkpoint_filenames = checkpoint_filenames[index:]
    
    tensor_board_logger = TensorBoardLogger(
        save_dir=logs_root,
        version=version,
        name=experiment_name,
        sub_dir='val'
    )
    workspace = Workspace(
        devices=devices,
        precision=precision,
        val_check_interval=0,
        logger=tensor_board_logger,
        deterministic=True
    )
    workspace.initialize_workspace(globals(), locals())
    workspace.set_workspace(preparation, finishing)
    best_metrics = None
    best_metrics_filename = {}
    best_metrics_all = {}
    checkpoint_filenames_length = len(checkpoint_filenames)
    for idx, ckpt_filename in enumerate(checkpoint_filenames):
        tqdm.write(f'\n\033[4m\033[3m\033[1mThis is the information for task {idx + 1} out of {checkpoint_filenames_length}.\033[0m')
        runner.need_validation_results(need_images=need_images)
        workspace.validate(
            runner,
            val_data_loader,
            os.path.join(checkpoints_dir, ckpt_filename)
        )
        metrics = runner.take_validation_results()
        if output_root is not None:
            output_dir = os.path.join(output_root, experiment_name, version)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(metrics, os.path.join(output_dir, f'{ckpt_filename}.pt'))
        metrics = {key : value.mean().item() for key, value in metrics.items()}
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
        others = best_metrics_all[best_key]
        if len(others) > 0:
            out_str = '\tOthers - '
            for i, (key, value) in enumerate(others.items()):
                out_str += f'{key}: {value}'
                if i != len(others) - 1:
                    out_str += ', '
            tqdm.write(out_str)


def _test_show_metrics(metrics, trainer: pl.Trainer):
    trainer_metrics = trainer.callback_metrics
    num_metrics = len(metrics)
    out_str = ''
    for i, (key, value) in enumerate(metrics.items()):
        if key not in trainer_metrics:
            out_str += f'{key}: {value.mean().item()}'
            if i != num_metrics - 1:
                out_str += ', '
    if out_str != '':
        out_str = f'\n{out_str}'
    tqdm.write(out_str)


def test(
    runner: ModelRunner,
    test_data_loader: DataLoader,
    checkpoint_path: str,
    devices: Union[List[int], str, int] = 1,
    precision: str = 'bf16-mixed',
    output_root: Optional[str] = None,
    need_images: bool = True,
    logs_root: str = './logs',
    experiment_name: str = 'default',
    version: Union[int, str, None] = '',
    preparation: Optional[Callable[..., Any]] = None,
    finishing: Optional[Callable[..., Any]] = None
):
    """Tests the diffusion model.

    Args:
        runner (ModelRunner): The instance of :class:`ModelRunner`.
        test_data_loader (DataLoader): The data loader for test.
        checkpoint_path (str): The checkpoint path that is expected to be tested.
        devices: The devices to use. Can be set to a positive number (int or str),
            a sequence of device indices.
        precision (str): The precision label about this execution.
        output_root (Optional[str]): The root of the output generations and original
            targets. If ``None``, they will not be output.
        need_images (bool) : The flag that controls whether to use the validation images.
        logs_root (str): The root that the log files will be located in.
        experiment_name (str): The name of this experiment.
        version (Union[int, str, None]): The version of this experiment, which can be
            ``int`` or ``str``, ``""`` (empty string) means providing no version.
            If ``None``, the runner will infer the next version of this experiment from
            the experiment directory.
        preparation (Optional[Callable[..., Any]]): The function that will be called before
            any instructions.
        finishing (Optional[Callable[..., Any]]): The function that will be called after
            any instructions.
    
    """
    tensor_board_logger = TensorBoardLogger(
        save_dir=logs_root,
        version=version,
        name=experiment_name,
        sub_dir='test'
    )
    workspace = Workspace(
        devices=devices,
        precision=precision,
        val_check_interval=0,
        logger=tensor_board_logger,
        deterministic=True
    )
    workspace.initialize_workspace(globals(), locals())
    workspace.set_workspace(preparation, finishing)
    runner.need_test_results(need_images=need_images)
    workspace.test(
        runner,
        test_data_loader,
        checkpoint_path
    )
    metrics = runner.take_test_results()
    
    if output_root is not None:
        output_dir = os.path.join(output_root, experiment_name, version)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(metrics, os.path.join(output_dir, 'test_results.pt'))
    
    _test_show_metrics(metrics, workspace)
    
    # metrics, generation, target = metrics
    # _test_show_metrics(metrics, trainer)
    
    # comparison = {
    #     'generation' : generation.permute(0, 2, 3, 1).cpu(),
    #     'target' : target.permute(0, 2, 3, 1).cpu(),
    #     'error': torch.abs(generation - target).permute(0, 2, 3, 1).cpu()
    # }
    
    # if output_root is not None:
    #     output_dir = os.path.join(output_root, experiment_name, version)
    #     os.makedirs(output_dir, exist_ok=True)
    #     torch.save(comparison, os.path.join(output_dir, 'test_collection.pt'))
    
    # tqdm.write("Start logging results... If you do not want to skip, press CTRL + C")
    # _tensorboard_vis(
    #     tensor_board_logger.experiment,
    #     comparison,
    #     metrics,
    #     {'generation' : 'gray', 'target' : 'gray', 'error': 'Reds'}
    # )