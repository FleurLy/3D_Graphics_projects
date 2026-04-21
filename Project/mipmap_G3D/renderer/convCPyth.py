#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chargement des fonctions C de downsampling mipmap via ctypes.
Les .so sont dans le sous-dossier c-so/ (compiler avec make).
"""
import ctypes
import os
import sys
from numpy.ctypeslib import ndpointer


def _load(name):
    ext = 'dll' if sys.platform.startswith('win') else 'so'
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'c-so', f'{name}.{ext}')
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Librairie introuvable : {lib_path}\n"
            f"Compilez avec :  cd renderer/c-so && make"
        )
    return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)


def boxPyth():
    """box_downsample(img, H, W, out) — moyenne 2x2, RGB float32."""
    dll  = _load('box')
    func = dll.box_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img  H*W*3
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out  (H/2)*(W/2)*3
    ]
    return func


def gaussianPyth():
    """gaussian_downsample(img, H, W, out) — noyau gaussien 4-taps, RGB float32."""
    dll  = _load('gaussian_downsample')
    func = dll.gaussian_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out
    ]
    return func


def lanczosKernelPyth():
    """lanczos_kernel_vals(x, n, a, result) — noyau Lanczos scalaire."""
    dll  = _load('lanczos_kernel_vals')
    func = dll.lanczos_kernel_vals
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # x      n valeurs
        ctypes.c_int,                                      # n
        ctypes.c_int,                                      # a
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # result n valeurs
    ]
    return func


def lanczosPyth():
    """lanczos_downsample(img, H, W, a, out) — filtre Lanczos ordre a, RGB float32."""
    dll  = _load('lanczos_downsample')
    func = dll.lanczos_downsample
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
    """med_downsample(img, H, W, out) — mediane 2x2, RGB float32."""
    dll  = _load('med')
    func = dll.med_downsample
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # img  H*W*3
        ctypes.c_int,                                      # H
        ctypes.c_int,                                      # W
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # out  (H/2)*(W/2)*3
    ]
    return func
