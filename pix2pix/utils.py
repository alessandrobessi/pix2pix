import os
import time
from typing import Tuple


def create_working_env() -> Tuple[str, str, str, str]:
    for d in ('runs', 'logs', 'examples'):
        if not os.path.exists(os.path.join(os.getcwd(), d)):
            os.makedirs(os.path.join(os.getcwd(), d))

    now = str(int(time.time()))

    runs_dir = os.path.join(os.getcwd(), 'runs', now)
    os.makedirs(runs_dir)

    logs_dir = os.path.join(os.getcwd(), 'logs', now)
    os.makedirs(logs_dir)

    examples_dir = os.path.join(os.getcwd(), 'examples', now)
    os.makedirs(examples_dir)

    data_dir = os.path.join(os.getcwd(), 'data')

    return data_dir, runs_dir, logs_dir, examples_dir
