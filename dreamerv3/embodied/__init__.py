try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

from .core import *
from . import envs, replay

try:
    from . import run
except ModuleNotFoundError as e:
    if e.name != "jax":
        raise
    run = None
