from typing import Sequence
import os


def set_devices(devices):
    if devices is not None:
        if isinstance(devices, Sequence):
            if len(devices) == 1:
                devices = devices[0]
                devices_string = f'{devices}'
            else:
                devices_string = ",".join(map(str, devices))
        elif isinstance(devices, int):
            devices_string = f'{devices}'
        os.environ['CUDA_VISIBLE_DEVICES'] = devices_string
    return devices