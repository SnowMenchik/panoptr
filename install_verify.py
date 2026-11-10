"""Verify dependencies in the candidate environment before it is activated."""
import sys
import cv2
import numpy
import torch
import torchvision
import ultralytics
import PySide6.QtWidgets
import psutil
import lap
import yaml
import scipy
import matplotlib
if sys.platform == "win32":
    import pygrabber
    import comtypes
if not hasattr(cv2, 'VideoCapture'):
    raise RuntimeError('OpenCV has no VideoCapture')
print('Python:', sys.executable)
print('Python version:', sys.version.split()[0])
print('OpenCV:', cv2.__version__, cv2.__file__)
print('Core dependency imports: OK')
