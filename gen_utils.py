import subprocess
from multiprocessing import Pool
from tqdm import tqdm


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
    if 'Exception' in out:
        raise Exception('Error returned: \n{}'.format(out))
    return out


def run_and_print(command):
    print("##Running command: " + command + "##")
    out = run_bash_command(command)
    print(out, flush=True)


def list_multiprocessing(param_lst, func, **kwargs):
    workers = kwargs.pop('workers')
    with Pool(workers) as p:
        apply_lst = [([params], func, i, kwargs) for i, params in enumerate(param_lst)]
        result = list(tqdm(p.imap(_apply_lst, apply_lst), total=len(apply_lst)))
    return [_[1] for _ in result]
