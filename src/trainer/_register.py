from util.register import REGISTER
from .hclr_trainer import HCLRTrainer


TRAINERS = REGISTER("trainers")
TRAINERS.register_module(HCLRTrainer)
