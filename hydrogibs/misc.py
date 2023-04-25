import numpy as np
from matplotlib import pyplot as plt
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


class SoCoSe:

    def __init__(self,
                 S: float,
                 L: float,
                 Pa: float,
                 P10: float,
                 ta: float,
                 Montana2: float,
                 zeta: float = 1.0,
                 tf: float = 5,
                 dt: float = 0.01) -> None:

        self.Pa = Pa
        self.P10 = P10

        self.ds = -0.69 + 0.32*np.log(S) + 2.2*np.sqrt(Pa/(P10*ta))
        self.J = 260 + 21*np.log(S/L) - 54*np.sqrt(Pa/P10)

        k = 24**Montana2/21*P10/(1 + np.sqrt(S)/(30*self.ds**(1/3)))
        rho = 1 - 0.2 * self.J / (k * (1.25*self.ds)**(1-Montana2))
        self.Q10 = zeta * k*S * rho**2 / ((15-12*rho)*(1.25*self.ds)**Montana2)

        self.time = np.arange(0, tf, step=dt)
        tau = 2*self.time/(3*self.ds)
        self.Q = self.Q10 * 2 * tau**4 / (1 + tau**8)


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
        warn(f"\ncrupedix: Catchment area is not within recommended range: "
             f"{S:.3e} not in [1,4 * 10^3 km^2 - 52 * 10^3 km^2]")
    return R * S**0.8 * (Pj10/80)**2


class QdF:

    """
    Based on rainfall GradEx,
    can estimate discharges for catchments of model type:
        - Soyans
        - Florac
        - Vandenesse

    Args:
        model (str):          Either 'Soyans', 'Florac' or 'Vandenesse'
        ds    (float) [h]:    Specific duration
        S     (float) [km^2]: Catchment surface
        L     (float) [km]:   Length of the thalweg
        im    (float) [%]:    Mean slope of the thalweg

    Calculates:
        tc (float) [h]: concentration time
    """

    _coefs = dict(

        soyans=dict(
            A=(2.57, 4.86, 0),
            B=(2.10, 2.10, 0.050),
            C=(1.49, 0.660, 0.017)),

        florac=dict(
            A=(3.05, 3.53, 0),
            B=(2.13, 2.96, 0.010),
            C=(2.78, 1.77, 0.040)),

        vandenesse=dict(
            A=(3.970, 6.48, 0.010),
            B=(1.910, 1.910, 0.097),
            C=(3.674, 1.774, 0.013))
    )

    def __init__(self, model, ds, S, L, im) -> None:
        """
        Based on rainfall GradEx,
        can estimate discharges for catchments of model type:
            - Soyans
            - Florac
            - Vandenesse

        Args:
            model (str):          Either 'Soyans', 'Florac' or 'Vandenesse'
            ds    (float) [h]:    Specific duration
            S     (float) [km^2]: Catchment surface
            L     (float) [km]:   Length of the thalweg
            im    (float) [%]:    Mean slope of the thalweg

        Calculates:
            tc (float) [h]: concentration time
        """
        self.coefs = self._coefs[model]
        self.ds = ds
        self.im = im
        self.S = S
        self.L = L

        self.tc = Turraza(S, L, im)

    def _calc_coefs(self, a):
        a1, a2, a3 = a
        return 1/(a1*self._d/self.ds + a2) + a3

    def discharge(self, d, T, Qsp, Q10):
        """
        Estimates the discharge for a certain flood duration
        and a certain return period

        Args:
            d (numpy.ndarray) [h]: duration of the flood
            T (numpy.ndarray) [y]: Return period
            Qsp (numpy.ndarray): Specific discharge
            Q10 (numpy.ndarray): Discharge for return period of 10 years

        Returns:
            (numpy.ndarray): Flood discharge
        """

        self._d = np.asarray(d)
        Qsp = np.asarray(Qsp)
        Q10 = np.asarray(Q10)
        T = np.asarray(T)

        self.A, self.B, self.C = map(
            self._calc_coefs,
            self.coefs.values()
        )
        return Q10 + Qsp * self.C * np.log(1 + self.A * (T-10)/(10*self.C))


def rationalMethod(S: float,
                   Cr: float,
                   tc: float,
                   ip: float = 1.0,
                   dt: float = 0.01) -> tuple:
    """
    Computes a triangular hydrogram from a flood with volume Cr*tc*S

    Args:
        S (float): Catchemnt area
        Cr (float): Peak runoff coefficient
        tc (float): Concentration time
        ip (float) [mm/h]: Rainfall intensity
        dt (float): timestep, default to 1 if not specified

    Returns:
        time [h], discharge [m^3/s] (numpy.ndarray, numpy.ndarray)
    """

    q = Cr*ip*S
    Qp = q/3.6

    time = np.arange(0, 2*tc, step=dt)
    Q = np.array([
        Qp * t/tc if t < tc else Qp * (2 - t/tc)
        for t in time
    ])

    return time, Q


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


def QdF_main():
    qdf = QdF(model="soyans", ds=1, S=1.8, L=2, im=25)
    Q10 = crupedix(S=1.8, Pj10=72, R=1.75)
    d = np.linspace(0, 3)
    Q100 = qdf.discharge(d, 100, Q10, Q10)
    plt.plot(d, Q100)
    plt.show()
