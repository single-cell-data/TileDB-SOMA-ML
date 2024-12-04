import json
import shlex
from functools import partial
from os import makedirs
from os.path import join
from subprocess import check_call
from sys import stderr
from tempfile import TemporaryDirectory

from click import group, argument, option


err = partial(print, file=stderr)


@group
def tdbsml():
    pass


@tdbsml.command
@option('-l', '--use-lightning', is_flag=True, help="Use PyTorch Lightning for training")
@option('-t', '--tissue', help='"tissue_general" filter to pass to obs_value_filter')
@option('-w', '--workers', type=int, help="Number of GPU workers to use")
@argument('in_nb_path')
@argument('out_nb_path', required=False)
def benchmark(use_lightning, tissue, workers, in_nb_path, out_nb_path):
    if not out_nb_path:
        name = "lightning" if use_lightning else "torch"
        if workers is not None:
            name += f"{workers}"
        makedirs(tissue, exist_ok=True)
        out_nb_path = f"{tissue}/{name}.ipynb"
        err(f"Using {out_nb_path=}")

    with TemporaryDirectory() as tmpdir:
        tmpfile = join(tmpdir, "time.log")
        cmd = [
            "/usr/bin/time", "-v", "-o", tmpfile,
            "papermill",
            "-p", "tissue", tissue,
            *(["-p", "lightning", f"{use_lightning}"] if use_lightning else []),
            *([] if workers is None else ["-p", "workers", f"{workers}"]),
            in_nb_path,
            out_nb_path,
        ]
        err(f"Running: {shlex.join(cmd)}")
        check_call(cmd)
        with open(tmpfile, "r") as fd:
            time_lines = list(fd)

    with open(out_nb_path, 'r') as nb_fd:
        nb = json.load(nb_fd)

    time_cell = {
        "cell_type": "markdown",
        "source": time_lines,
        "metadata": {},
    }
    nb['cells'].append(time_cell)
    with open(out_nb_path, 'w') as fd:
        json.dump(nb, fd, indent=1, ensure_ascii=False)
