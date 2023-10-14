import pytorch_lightning as pl
from pytorch_lightning.cli import LightningArgumentParser

from src.data import DataModule
from model_timm import TimmClassificationModel

import torch
torch.set_float32_matmul_precision('medium')


model_class = TimmClassificationModel
dm_class = DataModule

parser = LightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data")
parser.add_argument(
    "--checkpoint", type=str, help="path to model checkpoint", required=True
)
args = parser.parse_args()
args["logger"] = False  # Disable logging

model = model_class.load_from_checkpoint(args["checkpoint"], weights=None)
dm = dm_class.load_from_checkpoint(args["checkpoint"])

trainer = pl.Trainer.from_argparse_args(args)
trainer.test(model, datamodule=dm)
