import os
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import time

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

root_dir = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


