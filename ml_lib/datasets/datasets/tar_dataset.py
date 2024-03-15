from typing import TypeVar, IO
from os import PathLike
import tarfile

from ..base_classes import Dataset
from ..datapoint import Datapoint, DictDatapoint
from ..registration import register



PointType = TypeVar("PointType", bound=Datapoint)

class TarDataset(Dataset[PointType]):
    """
    Reads the data from a tar archive file 
    (which is a nice way of storing your data in a "multiple files" style
     without actually needing file descriptors for every point,
     which can be kinda slow, or overloading your inodes, etc.)

    Q/A

    In what order does it retrieve the data?
        In the default order of tarfile.getmembers(). I think alphabetical order.

    Does it read all the data to RAM?
        No. But you can pipe it to CacheTransform if that's what you wish.

    Can I use several workers with this?:
        Yes

    What isn't safe?
        please don't edit the tar file while reading from it?
    """

    file_path: PathLike

    members_list: list[str]
    
    _file: None|tarfile.TarFile = None

    def __init__(self, tar_file: PathLike):
        self.file_path = tar_file
        self._file = None
        file = self.tar_file # actually initializes the _file
        self.members_list = file.getnames()

    def __getitem__(self, i):
        member_name = self.members_list[i]
        file = self.tar_file
        reader = file.extractfile(member_name)
        if reader is None:
            raise ValueError(f"member {member_name} in file {self.file_path} is not a file or a link. This is unsupported.")
        return self.read_element(reader)

    def __len__(self):
        return len(self.members_list)

    def read_element(self, file: IO[bytes]) -> PointType:
        del file
        raise NotImplementedError(f"you cannot use {self.__class__} directly, you need to overload  the read_element function that reads data from one of the files in the tarfile")

    @property
    def tar_file(self) -> tarfile.TarFile:
        if self._file is not None:
            return self._file
        file = tarfile.open(self.file_path)
        self._file = file
        return file

    def __getstate__(self):
        return dict(file_path=self.file_path, 
                    members_list = self.members_list)

    def __setstate__(self, d):
        self.file_path = d["file_path"]
        self.members_list = d["members_list"]
        
    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.file_path)})"

        
@register
class TarNpzDataset(TarDataset[DictDatapoint]):
    """
    Dataset that takes a tar file containing npz files 
    (compressed numpy tensors).
    The numpy vectors are transformed to torch tensors, and returned
    as DictDatapoints
    """

    def read_element(self, file: IO[bytes]):
        import numpy as np
        with np.load(file) as npz:
            result = {f: npz[f] for f in npz.files}
        return DictDatapoint(result)


class TarTorchDataset(TarDataset[DictDatapoint]):
    """
    Dataset that takes a tar file containing pt files 
    (pickle files saved with torch.save).
    The numpy vectors are transformed to torch tensors, and returned
    as DictDatapoints
    """

    def read_element(self, file: IO[bytes]):
        import torch
        result = torch.load(file)
        if not isinstance(result, dict):
            result = {"x": result}

        return DictDatapoint(result)
