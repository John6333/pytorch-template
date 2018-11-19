import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Logger']

class Logger(object):
    def __init__(self, fpath, title=None):
        self.fpath = fpath
        self.title=title

    def set_names(self, names):
        # initialize numbers as empty list
        self.names = names
        with open(self.fpath, 'w') as f:
            info = '\t'.join(self.names) + '\n'
            f.write(info)

    def append(self, info):
        with open(self.fpath, 'a') as f:
            f.write(info+'\n')
