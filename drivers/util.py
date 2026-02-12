import json
import logging
import os
from os import PathLike
import shlex
import signal
import subprocess
from subprocess import CompletedProcess
from typing import Optional


def all_equal(iterable) -> bool:
    """ Returns true if all values in iterable are equal """
    return len(set(iterable)) <= 1

def await_input(prompt: str, is_valid_input) -> str:
    """ Repeatedly ask the user for input until it is valid. """
    response = input(prompt)
    while not is_valid_input(response):
        response = input(prompt)
    return response

def load_json(fpath: PathLike) -> dict:
    """ Load the given json file into a dict """
    with open(fpath, "r") as fp:
        return json.load(fp)

def mean(iterable) -> float:
    """ Returns the mean of the given iterable """
    if not hasattr(iterable, "__len__"):
        iterable = list(iterable)
    return sum(iterable) / len(iterable) if len(iterable) > 0 else 0

def run_command(cmd: str, timeout: Optional[int] = None, dry: bool = False) -> CompletedProcess:
    logging.debug(f"Running command: {cmd}")
    if dry:
        return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    
    cmd_args = shlex.split(cmd)
    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return CompletedProcess(args=cmd, returncode=process.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        raise


def _kill_process_tree(process: subprocess.Popen):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            process.kill()
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        process.communicate()
