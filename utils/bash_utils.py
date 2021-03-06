import logging
import subprocess
import sys
from multiprocessing import Pool
from tqdm import tqdm

from time import time
from collections import defaultdict

timings = defaultdict(list)


def _apply_lst(args):
    params, func, num, kwargs = args
    return num, func(*params, **kwargs)


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)

    return iter(p.stdout.readline, b'')


def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)

    out = p.communicate()[0].decode('utf-8')
    if 'Exception' in out or 'Error' in out:
        raise Exception('Error returned: \n{}'.format(out))
    return out


def run_and_print(command, command_name=None):
    logger = logging.getLogger(sys.argv[0])
    name = command_name.rstrip() if command_name is not None else 'command'
    print_string = f'Running {name}: \n{command}'
    logger.info(print_string)

    start = time()

    out = run_bash_command(command)

    run_time = time() - start
    timings[command_name].append(run_time)

    if len(out) > 0:
        output_string = name + ' output:'
        print(output_string, out, sep='\n', flush=True)


def list_multiprocessing(params_list, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers) as p:
        apply_lst = [((params if isinstance(params, list) else [params]), func, i, kwargs)
                     for i, params in enumerate(params_list)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [item[1] for item in result]
