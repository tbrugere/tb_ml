from torch.utils.data import Dataset

from .base_classes import Transform
from ..register import Register

register : Register[Dataset]= Register(Dataset)

transform_register : Register[Transform] = Register(Transform)
