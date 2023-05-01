import numpy as np
from scipy.optimize import OptimizeResult, minimize
from warnings import warn


g = 9.81


def montana(
        d: np.ndarray,
        a: float,
        b: np.ndarray,
        cum: bool = True
) -> np.ndarray:
    """
    Relation between duration and rainfall

    Args:
        d (numpy.ndarray): rainfall duration
        a (float): first Montana coefficient
        b (float): second Montana coefficient
    Returns:
        P (numpy.ndarray): rainfall (cumulative if cum=True, intensive if not)
    """
    d = np.asarray(d)
    return a * d**-b if cum else a * d**(1-b)


def montana_inv(
        P: np.ndarray,
        a: float,
        b: np.ndarray,
        cum: bool = True
) -> np.ndarray:
    """
    Relation between rainfall and duration

    Args:
        I (numpy.ndarray): rainfall (cumulative if cum=True, intensity if not)
        a (float): first Montana coefficient
        b (float): second Montana coefficient
    Returns:
        d (numpy.ndarray): rainfall duration
    """
    P = np.asarray(P)
    return (P/a)**(-1/b) if cum else (P/a)**(1/(1-b))


def fit_montana(d: np.ndarray,
                P: np.ndarray,
                a0: float = 40,
                b0: float = 1.5,
                cum=True,
                tol=0.1) -> OptimizeResult:
    """
    Estimates the parameters for the Monatana law
    from a duration array and a Rainfall array

    Args:
        d (numpy.ndarray): event duration array
        P (numpy.ndarray): rainfall (cumulative if cum=True, intensive if not)
        a0 (float): initial first montana coefficient for numerical solving
        b0 (float): initial second montana coefficient for numerical solving

    Returns:
        res (OptimizeResult): containing all information about the fitting,
                              access result via attribute 'x',
                              access error via attribute 'fun'
    """

    d = np.asarray(d)
    P = np.asarray(P)

    res = minimize(
        fun=lambda M: np.linalg.norm(P - montana(d, *M, cum)),
        x0=(a0, b0),
        tol=tol
    )

    if not res.success:
        warn(f"fit_montana: {res.message}")

    return res


def thalweg_slope(lk, ik, L):
    """
    Weighted avergage thalweg slope [%]

    Args:
        lk (numpy.ndarray): length of k-th segment
        ik (numpy.ndarray) [%]: slope of the k-th segment

    Returns:
        im (numpy.ndarray) [%]: thalweg slope
    """
    lk = np.asarray(lk)
    ik = np.asarray(ik)
    return (
        L / (lk / np.sqrt(ik)).sum()
    )**2


def Turraza(S, L, im):
    """
    Empirical estimation of the concentration time of a catchment

    Args:
        S (float) [km^2]: Catchment area
        L (float) [km]: Longest hydraulic path's length
        im (float) [%]: weighted average thalweg slope,
                        should be according to 'thalweg_slope' function

    Returns:
        tc (float) [h]: concentration time
    """
    return 0.108*np.sqrt((S*L)**3/im)


def specific_duration(S: np.ndarray) -> np.ndarray:
    """
    Returns duration during which the discharge is more than half its maximum.
    This uses an empirical formulation.
    Unrecommended values will send warnings.

    Args:
        S (float | array-like) [km^2]: Catchment area

    Returns:
        ds (float | array-like) [?]: specific duration
    """

    _float = isinstance(S, float)
    S = np.asarray(S)

    ds = np.exp(0.375*S + 3.729)/60  # TODO seconds or minutes?

    if not 10**-2 <= S.all() <= 15:
        warn(f"Catchment area is not within recommended range [0.01, 15] km^2")
    elif not 4 <= ds.all() <= 300:
        warn(f"Specific duration is not within recommended range [4, 300] mn")
    return float(ds) if _float else ds


def crupedix(S: float, Pj10: float, R: float = 1.0):
    """
    Calculates the peak flow Q10 from a daily rain of 10 years return period.

    Args:
        S (float) [km^2]: catchment area
        Pj10 (float) [mm]: total daily rain with return period of 10 years
        R (float) [-]: regionnal coefficient, default to 1 if not specified

    Returns:
        Q10 (float): peak discharge flow for return period T = 10 years
    """
    if not 1.4 <= S <= 52*1000:
        warn(f"\ncrupedix: Catchment area is not within recommended range:\n\t"
             f"{S:.3e} not in [1,4 * 10^3 km^2 - 52 * 10^3 km^2]")
    return R * S**0.8 * (Pj10/80)**2


def zeller(montana_params: tuple,
           duration: float,
           vtime: float,  # TODO
           rtime: float,  # TODO
           atol: float = 0.5) -> None:

    P = montana(duration, *montana_params)
    Q = P/vtime

    if not np.isclose(vtime + rtime, duration, atol=atol):
        warn(f"\nt_v and t_r are not close enough")
    return Q
