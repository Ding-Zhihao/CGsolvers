import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import loadmat, savemat
from csr_convert import *

A = loadmat('../data/input')['A']
A_csr = to_csr(A)

