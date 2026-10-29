"""Checks on estimated parameters shared by profiles and reports."""
from scipy.optimize import minimize_scalar

from .spec import CANON


def probability_shape(method, parameter):
    """(monotone, minimum of the TK derivative numerator) of a weighting function.
    TK weighting is monotone only for gamma >= 0.2792."""
    if parameter is None or parameter <= 0: return False, None
    if method == "prelec": return True, None
    if method != "tk": return None, None
    g = parameter
    if g == 1: return True, 1.0
    def numerator(p):
        return (g-1)*p**g + (1-p)**(g-1)*(g+(1-g)*p)
    minimum = min(g, minimize_scalar(numerator, bounds=(1e-14, 1-1e-14), method="bounded").fun)
    return bool(minimum >= -1e-12), float(minimum)


def probability_valid(result):
    """True if every cluster's probability weighting function is monotone."""
    method = result["spec"].pw
    position = CANON[method]["PALPHA" if method == "prelec" else "GAM"]
    return all(probability_shape(method, float(theta[position]))[0] for theta in result["thetas"])
