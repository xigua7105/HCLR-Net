from util.register import REGISTER
from .ir_tester import IRTester

TESTER = REGISTER("trainers")
TESTER.register_module(IRTester)
