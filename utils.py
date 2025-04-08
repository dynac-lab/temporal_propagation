import re
from fractions import Fraction
from scipy.signal import resample_poly
import numpy as np

def resample(x, sr1, sr2, axis=0):
    '''sr1: target, sr2: source'''
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)
