import pandas as pd
import six


def order(frame, var):
    """Order a dataframe"""
    if isinstance(var, six.string_types):
        var = [var]  #let the command take a string or list
    varlist = [w for w in frame.columns if w not in var]
    frame = frame[var + varlist]
    return frame
