import os
import time
from typing import Tuple


def create_working_env() -> Tuple[str, str]:
    if not os.path.exists(os.path.join(os.getcwd(), 'runs')):
        os.makedirs(os.path.join(os.getcwd(), 'runs'))

    now = str(int(time.time()))

    runs_dir = os.path.join(os.getcwd(), 'runs', now)
    os.makedirs(runs_dir)

    logs_dir = os.path.join(os.getcwd(), 'logs', now)
    os.makedirs(logs_dir)

    return runs_dir, logs_dir
