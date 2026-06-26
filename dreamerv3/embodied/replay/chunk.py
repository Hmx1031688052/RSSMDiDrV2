import io
import pathlib
import shutil
import tempfile
import zipfile
from datetime import datetime

import embodied
import numpy as np


class Chunk:
    def __init__(self, size, successor=None):
        now = datetime.now()
        self.time = now.strftime("%Y%m%dT%H%M%S") + f"F{now.microsecond:06d}"
        self.uuid = str(embodied.uuid())
        self.successor = successor
        self.size = size
        self.data = None
        self.length = 0

    def __repr__(self):
        succ = self.successor or str(embodied.uuid(0))
        succ = succ.uuid if isinstance(succ, type(self)) else succ
        return f"Chunk(uuid={self.uuid}, " f"succ={succ}, " f"len={self.length})"

    def __len__(self):
        return self.length

    def __bool__(self):
        return True

    def append(self, step):
        if not self.data:
            example = {k: embodied.convert(v) for k, v in step.items()}
            self.data = {k: np.empty((self.size,) + v.shape, v.dtype) for k, v in example.items()}
        for key, value in step.items():
            self.data[key][self.length] = value
        self.length += 1

    def save(self, directory):
        succ = self.successor or str(embodied.uuid(0))
        succ = succ.uuid if isinstance(succ, type(self)) else succ
        filename = f"{self.time}-{self.uuid}-{succ}-{self.length}.npz"
        filename = embodied.Path(directory) / filename
        data = {k: embodied.convert(v) for k, v in self.data.items()}
        with io.BytesIO() as stream:
            np.savez_compressed(stream, **data)
            stream.seek(0)
            filename.write(stream.read(), mode="wb")
        print(f"Saved chunk: {filename.name}")

    @classmethod
    def load(cls, filename):
        length = int(filename.stem.split("-")[3])
        with embodied.Path(filename).open("rb") as f:
            data = np.load(f)
            data = {k: data[k] for k in data.keys()}
        chunk = cls(length)
        chunk.time = filename.stem.split("-")[0]
        chunk.uuid = filename.stem.split("-")[1]
        chunk.successor = filename.stem.split("-")[2]
        chunk.length = length
        chunk.data = data
        return chunk

    @classmethod
    def scan(cls, directory, capacity=None, shorten=0):
        directory = embodied.Path(directory)
        filenames, total = [], 0
        for filename in reversed(sorted(directory.glob("*.npz"))):
            if capacity and total >= capacity:
                break
            filenames.append(filename)
            total += max(0, int(filename.stem.split("-")[3]) - shorten)
        return sorted(filenames)


class DiskChunk:
    """Replay chunk buffered as per-key memmaps to avoid large RAM spikes."""

    def __init__(self, size, successor=None, tempdir=None):
        now = datetime.now()
        self.time = now.strftime("%Y%m%dT%H%M%S") + f"F{now.microsecond:06d}"
        self.uuid = str(embodied.uuid())
        self.successor = successor
        self.size = size
        self.tempdir = pathlib.Path(str(tempdir or tempfile.gettempdir()))
        self.data = None
        self.paths = {}
        self.length = 0
        self.workdir = None

    def __repr__(self):
        succ = self.successor or str(embodied.uuid(0))
        succ = succ.uuid if hasattr(succ, "uuid") else succ
        return f"DiskChunk(uuid={self.uuid}, " f"succ={succ}, " f"len={self.length})"

    def __len__(self):
        return self.length

    def __bool__(self):
        return True

    def append(self, step):
        if self.data is None:
            self.tempdir.mkdir(parents=True, exist_ok=True)
            self.workdir = pathlib.Path(tempfile.mkdtemp(prefix=f"chunk-{self.uuid}-", dir=str(self.tempdir)))
            example = {k: embodied.convert(v) for k, v in step.items()}
            self.data = {}
            for index, (key, value) in enumerate(example.items()):
                path = self.workdir / f"{index:06d}.npy"
                shape = (self.size,) + value.shape
                self.data[key] = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=shape)
                self.paths[key] = path
        for key, value in step.items():
            self.data[key][self.length] = value
        self.length += 1

    def save(self, directory):
        succ = self.successor or str(embodied.uuid(0))
        succ = succ.uuid if hasattr(succ, "uuid") else succ
        filename = f"{self.time}-{self.uuid}-{succ}-{self.length}.npz"
        filename = embodied.Path(directory) / filename
        try:
            with zipfile.ZipFile(str(filename), mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for key, value in self.data.items():
                    value.flush()
                    with archive.open(f"{key}.npy", mode="w", force_zip64=True) as stream:
                        np.lib.format.write_array(stream, value[: self.length], allow_pickle=False)
            print(f"Saved chunk: {filename.name}")
        finally:
            self.close()

    def close(self):
        if self.data is not None:
            for value in self.data.values():
                try:
                    value.flush()
                except Exception:
                    pass
                mmap = getattr(value, "_mmap", None)
                if mmap is not None:
                    try:
                        mmap.close()
                    except Exception:
                        pass
            self.data = None
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None
