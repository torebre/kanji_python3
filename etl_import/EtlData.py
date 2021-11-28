import string
from dataclasses import dataclass


@dataclass
class EtlData:
    name: string
    path: string
    struct_format: string
