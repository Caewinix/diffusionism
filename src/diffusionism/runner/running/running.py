from typing import Union, Literal, Callable, Optional, List, Sequence, Any
from typing_extensions import override
import os
import warnings
from datetime import timedelta
from copy import copy
import re
import bisect
from tqdm import tqdm
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint as IndistinguishableModelCheckpoint
from pytorch_lightning.trainer.states import RunningStage
from lightning_fabric.utilities.types import _PATH
from ...utils.workspace import Workspace
from ..model_runner import ModelRunner, params_t


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
    
    check_val_every_n_epoch = None
    val_check_interval = None
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
        if every_n_epochs is None:
            val_check_interval = every_n_train_steps
        else:
            check_val_every_n_epoch = every_n_epochs
        val_checkpoint_callback = ModelCheckpoint(
            name='val_checkpoint',
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
        check_val_every_n_epoch = 1
        limit_val_batches = 0
    if saving_step_interval is not None or saving_epoch_interval is not None:
        every_n_train_steps, every_n_epochs = _distinguish_step_and_epoch(
            saving_step_interval,
            saving_epoch_interval,
            'saving_step_interval',
            'saving_epoch_interval'
        )
        saving_checkpoint_callback = ModelCheckpoint(
            name='saving_checkpoint',
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
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        val_check_interval=val_check_interval,
        logger=tensor_board_logger,
        deterministic=True,
        num_sanity_val_steps=None if has_trained else 0
    )
    workspace.initialize_workspace(globals(), locals())
    workspace.set_workshop(preparation, finishing)
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
    devices_more_than_one = _check_devices_more_than_one(devices)
    
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
        deterministic=True,
        enable_checkpointing=False
    )
    workspace.initialize_workspace(globals(), locals())
    workspace.set_workshop(preparation, finishing)
    
    _ignore_checkpoint_warnings(workspace)
    if devices_more_than_one:
        _ignore_data_connector_warnings(workspace)
    
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
    
    _restore_type(workspace._checkpoint_connector, _WarningsIgnoredCheckpointConnector)
    if devices_more_than_one:
        _restore_type(workspace._data_connector, _WarningsIgnoredDataConnector)


def test(
    runner: ModelRunner,
    test_data_loader: DataLoader,
    checkpoint_path: str,
    devices: Union[List[int], str, int] = 1,
    precision: str = 'bf16-mixed',
    output_root: Optional[str] = None,
    output_filename: Optional[str] = 'test_results.pt',
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
        output_filename (Optional[str]): The filename of the output result.
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
    devices_more_than_one = _check_devices_more_than_one(devices)
    
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
        deterministic=True,
        enable_checkpointing=False
    )
    workspace.initialize_workspace(globals(), locals())
    workspace.set_workshop(preparation, finishing)
    runner.need_test_results(need_images=need_images)
    
    _ignore_checkpoint_warnings(workspace)
    if devices_more_than_one:
        _ignore_data_connector_warnings(workspace)
    
    workspace.test(
        runner,
        test_data_loader,
        checkpoint_path
    )
    metrics = runner.take_test_results()
    
    if output_root is not None:
        output_dir = os.path.join(output_root, experiment_name, version)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(metrics, os.path.join(output_dir, output_filename))
    
    # _test_show_metrics(metrics, workspace)
    
    _restore_type(workspace._checkpoint_connector, _WarningsIgnoredCheckpointConnector)
    if devices_more_than_one:
        _restore_type(workspace._data_connector, _WarningsIgnoredDataConnector)
    
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


def _check_devices_more_than_one(devices):
    if isinstance(devices, Sequence):
        num_devices = len(devices)
    elif isinstance(devices, int):
        num_devices = devices
    else:
        num_devices = 1
    return num_devices > 1 or num_devices == -1


class ModelCheckpoint(IndistinguishableModelCheckpoint):
    def __init__(
        self,
        name: str,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath, filename, monitor, verbose, save_last,
            save_top_k, save_weights_only, mode, auto_insert_metric_name,
            every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end,
            enable_version_counter
        )
        self.name = name
    
    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().setup(trainer, pl_module, stage)
    
    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
            name=self.name
        )


class TensorBoardLogger(pl_loggers.TensorBoardLogger):
    @property
    def log_dir(self) -> str:
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


def _find_type_attr(type: type, name: str):
    mro = type.__mro__
    for current_type in mro:
        if name in type.__dict__:
            return current_type


def _make_hook_type(hook_type: type):
    def __getattribute__(self, name):
        if name == '__class__' and _find_type_attr(type(self), '__class__') is None:
            return hook_type.__base__
        return super(type(self), self).__getattribute__(name)
    
    def __setattr__(self, name, value):
        if name == '__class__' and _find_type_attr(type(self), '__class__') is None:
            hook_type.__bases__ = (value,)
        else:
            super(type(self), self).__setattr__(name, value)
    
    hook_type.__getattribute__ = __getattribute__
    hook_type.__setattr__ = __setattr__
    return hook_type


def _pretend_hook_type(hook_type: type, reference_type: type):
    hook_type.__module__ = reference_type.__module__
    hook_type.__doc__ = reference_type.__doc__
    hook_type.__name__ = reference_type.__name__
    hook_type.__qualname__ = reference_type.__qualname__


class _WarningsIgnoredCheckpointConnector: ...
def _ignore_checkpoint_warnings(trainer: pl.Trainer):
    checkpoint_connector_type = type(trainer._checkpoint_connector)
    @_make_hook_type
    class CheckpointConnector(checkpoint_connector_type, _WarningsIgnoredCheckpointConnector):
        def restore_callbacks(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                super().restore_callbacks()
    _pretend_hook_type(CheckpointConnector, checkpoint_connector_type)
    trainer._checkpoint_connector.__class__ = CheckpointConnector


class _WarningsIgnoredDataConnector: ...
def _ignore_data_connector_warnings(trainer: pl.Trainer):
    data_connector_type = type(trainer._data_connector)
    @_make_hook_type
    class DataConnector(data_connector_type, _WarningsIgnoredDataConnector):
        def _resolve_sampler(
            self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage] = None
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                super()._resolve_sampler(dataloader, shuffle, RunningStage.PREDICTING)
    _pretend_hook_type(DataConnector, data_connector_type)
    trainer._data_connector.__class__ = DataConnector


def _restore_type(obj, reference_type):
    obj_type = type(obj)
    bases = obj_type.__bases__
    if issubclass(obj_type, reference_type):
        super(obj_type, obj).__setattr__('__class__', bases[0])
    else:
        all_bases = [(obj_type, base, base.__bases__) for base in bases]
        while len(all_bases) != 0:
            child, current_type, bases = all_bases.pop()
            if issubclass(current_type, reference_type):
                child.__bases__ = bases
                break
            all_bases.extend(((current_type, base, base.__bases__) for base in bases))


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


# def _test_show_metrics(metrics, trainer: pl.Trainer):
#     trainer_metrics = trainer.callback_metrics
#     num_metrics = len(metrics)
#     out_str = ''
#     for i, (key, value) in enumerate(metrics.items()):
#         if key not in trainer_metrics:
#             out_str += f'{key}: {value.mean().item()}'
#             if i != num_metrics - 1:
#                 out_str += ', '
#     if out_str != '':
#         out_str = f'\n{out_str}'
#     tqdm.write(out_str)