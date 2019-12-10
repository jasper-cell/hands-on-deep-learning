# coding: utf-8

%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

def use_svg_display():
    #用矢量图显示
    display.set_matplotlib_formats('svg')