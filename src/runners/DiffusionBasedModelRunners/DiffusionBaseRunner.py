from abc import ABC
from runners.BaseRunner import BaseRunner


class DiffusionBaseRunner(BaseRunner, ABC):
    def __init__(self, config):
        super().__init__(config)
