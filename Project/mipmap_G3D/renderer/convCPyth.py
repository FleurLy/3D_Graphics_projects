import ctypes
import os
import sys
from numpy.ctypeslib import ndpointer

def load(name):
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'c-so', f'{name}.{'so'}')
    return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)


def boxPyth():
    func = load('box').box_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img 
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out  
    ]
    return func


def gaussianPyth():
    func = load('gaussian_downsample').gaussian_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out
    ]
    return func

def lanczosPyth():
    func = load('lanczos_downsample').lanczos_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ctypes.c_int,                                      # a
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out
    ]
    return func

def medPyth():
    func = load('med').med_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out
    ]
    return func
