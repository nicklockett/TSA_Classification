from classes import *
from dataExtraction import *

bs = BodyScan("../data/data/a3d/955af05d26da95acc4e7c9821989e7e0.a3d")
bsg = BlockStreamGenerator(bs)
vals = bsg.generateStream()
print(vals)