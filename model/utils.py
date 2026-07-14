# =============================================================================
# Lead model conceptualization and development (theory, methodology, and
# implementation), software architecture & model coding (all components),
# data acquisition & processing, team supervision: Michael Miess
# Initial software architecture development, first coding, technical consulting:
# Joel Foramitti
# Lead of energy modeling: Ansir Ilyas
# Lead of water modeling: Dan Wang
# Model consulting & support: Asjad Naqvi
# Project conceptualization, funding & supervision: Yoshihide Wada
# =============================================================================

import numpy as np
import numpy.typing as npt
import datetime

NDArray = npt.NDArray[np.float64]


def step_array(v1: npt.ArrayLike, v2: npt.ArrayLike, T: int, q: int = 0):
    """
    Create an array of length T that jumps from v1 to v2 at step q.

    Can take any type of array as input, as long as it can be added together.
    The resulting array will have the time-step as the first index.
    """
    return np.stack([v1] * q + [v2] * (T-q))


def string_to_value(str, data):
    """Take a string like '-X' and return `-data['X']`"""
    if str == "" or str == "0":
        return 0
    elif "+" in str:
        return data[str[1:]]
    elif "-" in str:
        return - data[str[1:]]
    else:
        return data[str]


def libreoffice_int_to_datetime(days: int):
    """Converts a LibreOffice Calc date int to a Python datetime object."""
    # LibreOffice Calc's epoch date is December 30, 1899
    epoch_date = datetime.datetime(1899, 12, 30)
    return epoch_date + datetime.timedelta(days=days)


def to_float_or_nan(value):
    """Converts a string to a float, or NaN if it fails."""
    try:
        if type(value) is str:
            value = value.replace(',', '.')
        return float(value)
    except ValueError:
        return np.nan
