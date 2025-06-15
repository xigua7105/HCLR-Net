from util.register import REGISTER
from .hclrnet.model import HCLRNet


MODELS = REGISTER("models")
MODELS.register_module(HCLRNet)
