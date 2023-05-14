import numpy as np
from hydrogibs.misc import Turraza
from typing import Literal, Union
from matplotlib import pyplot as plt
from hydrogibs.ModelApp import ModelApp, Entry
from warnings import warn
from scipy.optimize import least_squares


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

    _coefs_threshold = dict(

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

    _coefs_mean = dict(

        soyans=dict(
            A=(0.87, 4.60, 0),
            B=(1.07, 2.50, 0.099),
            C=(0.569, 0.690, 0.046)),

        florac=dict(
            A=(1.12, 3.56, 0),
            B=(0.95, 3.18, 0.039),
            C=(1.56, 1.91, 0.085)),

        vandenesse=dict(
            A=(2.635, 6.19, 0.016),
            B=(1.045, 2.385, 0.172),
            C=(1.083, 1.75, 0))
    )

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
                 duration: Union[float, np.ndarray],
                 return_period: float,
                 specific_discharge: float,
                 discharge_Q10: float,
                 dt: float = None,
                 observation_time: float = None):

        self.duration = duration
        self.return_period = return_period
        self.specific_discharge = specific_discharge
        self.discharge_Q10 = discharge_Q10

        self.dt = dt if dt is not None else duration/100
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


def App(catchment: Catchment = None,
        rain: Rain = None,
        style: str = "seaborn",
        *args, **kwargs):
    if catchment is None:
        catchment = Catchment("soyans",
                              specific_duration=1,
                              surface=1.8,
                              length=2,
                              mean_slope=9.83/100)
    if rain is None:
        rain = Rain(np.linspace(0, 24), 100, 0.3, 0.3)

    if hasattr(catchment, "specific_duration"):
        entries = [("catchment", "specific_duration", "h", "ds")]
    else:
        entries = [
            ("catchment", "surface", "km^2", "S"),
            ("catchment", "length", "km", "L"),
            ("catchment", "mean_slope", "%", "im")
        ]
    entries += [
        ("rain", "return_period", "y", "T"),
        ("rain", "specific_discharge", "m3/s", "Qs"),
        ("rain", "discharge_Q10", "m3/s", "Q10"),
        ("rain", "tf", "h")
    ]
    entries = map(lambda e: Entry(*e), entries)
    ModelApp(
        catchment=catchment,
        rain=rain,
        entries=entries
    )


def qdf(catchment, rain):

    constants_threshold = list(
        catchment._coefs_threshold[catchment.model].values()
    )
    constants_mean = list(
        catchment._coefs_mean[catchment.model].values()
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


def main():

    App(catchment=Catchment(model="soyans",
                            specific_duration=1,
                            surface=1.8,
                            length=2,
                            mean_slope=9.83/100))


if __name__ == "__main__":
    main()
