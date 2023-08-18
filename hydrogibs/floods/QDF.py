import numpy as np
from ..misc.misc import Turraza
from typing import Literal
from matplotlib import pyplot as plt
from .constants import QDFcoefs_mean, QDFcoefs_threshold
from warnings import warn
from scipy.optimize import least_squares
from os.path import join, dirname


def _arange_QDFcoefs(path):
    with open(join(dirname(__file__), path)) as file:
        data = [
            line.split(',')
            for line in file.read().splitlines()[1:]
        ]
        data = [
            (line[0], *map(float, line[1:]))
            for line in data
        ]
    return {
        name: dict(A=alphas[:3], B=alphas[3:6], C=alphas[6:])
        for name, *alphas in data
    }


QDFcoefs_mean = _arange_QDFcoefs('qdf-mean.csv')
QDFcoefs_thres = _arange_QDFcoefs('qdf-thres.csv')


class Catchment:
    """
    Stores a QDF catchment's parameters.

    Creates a QDF event object when called with a QDF Rain object:
    >>> qdf = QDF(catchment, rain)
    Creates an Event object when applied to a Rain object
    >>> event = rain @ catchment

    Args:
        model              (str): The kind of river, possible choices are
                                    - 'soyans'
                                    - 'florac'
                                    - 'vandenesse'
        specific_duration (float) [h]:    Specific duration
        surface           (float) [km]:   Length of the thalweg
        length            (float) [%]:    Mean slope of the thalweg
        mean_slope        (float) [km^2]: Catchment surface
    """

    def __init__(self,
                 model: Literal["soyans", "florac", "vandenesse"],
                 specific_duration: float = None,
                 surface: float = None,
                 length: float = None,
                 mean_slope: float = None) -> None:

        self.model = model.lower()
        if specific_duration is not None:
            self.specific_duration = specific_duration
        else:
            self.surface = surface
            self.length = length
            self.mean_slope = mean_slope

    def __matmul__(self, rain):
        return rain @ self


class Rain:
    """
    Rain object to apply to a QDF Catchment object.

    Args:
        - time        (np.ndarray)       [h]
        - rain_func   (callable)   -> [mm/h]

    Creates a GR4h object when called with a Catchment object:
    >>> gr4h = GR4h(catchment, rain)
    Creates an Event object when applied to a catchment
    >>> event = rain @ catchment

    Args:

    """

    def __init__(self,
                 duration: float,
                 return_period: float,
                 specific_discharge: float,
                 discharge_Q10: float,
                 dt: float = .0,
                 observation_time: float = None):

        self.duration = duration
        self.return_period = return_period
        self.specific_discharge = specific_discharge
        self.discharge_Q10 = discharge_Q10
        self.dt = dt if dt else duration/100
        self.tf = (observation_time if observation_time is not None
                   else 5 * duration)

        assert 0 <= return_period
        assert 0 <= specific_discharge
        assert 0 <= discharge_Q10

    def __matmul__(self, catchment):
        return qdf(catchment=catchment, rain=self)


class Event:

    def __init__(self, time, discharge) -> None:

        self.time = time
        self.discharge = discharge

    def diagram(self, *args, **kwargs):
        return QDFdiagram(self, *args, **kwargs)


class QDFdiagram:

    def __init__(self,
                 event: Event,
                 style: str = "ggplot",
                 margin=0.1,
                 show=True) -> None:

        self.event = event
        self.margin = margin

        with plt.style.context(style):

            fig, ax = plt.subplots(figsize=(5, 3))
            self.line, = ax.plot(event.time, event.discharge)
            ax.set_xlabel("Time [h]")
            ax.set_ylabel("Discharge [m$^3$/s]")

            self.axes = (ax, )
            self.figure = fig
            plt.tight_layout()
            if show:
                plt.show()

    def update(self, event: Event):

        self.line.set_data(event.time, event.discharge)

    def zoom(self, canvas):

        ax = self.axes[0]
        ax.set_yscale("linear")
        t, Q = self.line.get_data()
        ylim = Q.max() * (1 + self.margin)
        ax.set_ylim((0, ylim if ylim else 1))
        ax.set_xlim((0, t.max()))
        canvas.draw()


def qdf(catchment, rain):

    constants_threshold = list(
        QDFcoefs_threshold[catchment.model.capitalize()].values()
    )
    constants_mean = list(
        QDFcoefs_mean[catchment.model.capitalize()].values()
    )

    if hasattr(catchment, "specific_duration"):
        ds = catchment.specific_duration
    else:
        ds = Turraza(
            catchment.surface,
            catchment.length,
            catchment.mean_slope
        )

    discharge_peak = discharge(Q10=rain.discharge_Q10,
                               Qsp=rain.specific_discharge,
                               T=rain.return_period,
                               constants=constants_mean,
                               d=rain.duration,
                               ds=ds)

    dt = rain.dt
    time = np.arange(0, rain.tf, step=rain.dt)
    Q = np.full_like(time, discharge_peak)

    ds = catchment.specific_duration
    i = time <= ds
    Q[i] = discharge_peak * time[i] / ds

    min_d = 0
    for i, t in enumerate(time[~i], start=i.sum()):

        result = least_squares(
            lambda d: discharge(Q10=rain.discharge_Q10,
                                Qsp=rain.specific_discharge,
                                T=rain.return_period,
                                constants=constants_threshold,
                                d=d,
                                ds=ds) - discharge_peak * (t - d)/ds,
            x0=min_d,
            bounds=(0, t)
        )

        d = float(result.x[0])
        q = discharge_peak * (t - d)/ds

        Q[i] = q
        min_d = d + dt

    return Event(time, Q)


def calc_coefs(constants, d, ds):
    return np.array([1/(a1*d/ds + a2) + a3
                     for a1, a2, a3 in constants])


def discharge(Q10, Qsp, T, constants, d, ds):
    A, B, C = calc_coefs(constants, d, ds)
    if 0.5 <= T <= 20:
        Q = A * np.log(T) + B
    elif T <= 1000:
        Q = C * np.log(1 + A * (T-10)/(10*C))
    else:
        warn(
            f"{T = :.0f} is not within [0.5:1000] years"
        )
    return Q10 + Qsp * Q
