from .base_classes import Transform, Dataset
from ..register import Register

register : Register[Dataset]= Register(Dataset)

transform_register : Register[Transform] = Register(Transform)
