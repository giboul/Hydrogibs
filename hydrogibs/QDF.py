import numpy as np
from hydrogibs.misc import Turraza
from typing import Literal
from matplotlib import pyplot as plt
from ModelApp import ModelApp, Entry


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

    def __init__(self,
                 model: Literal["soyans", "florac", "vandenesse"],
                 specific_duration: float = None,
                 surface: float = None,
                 length: float = None,
                 mean_slope: float = None) -> None:

        self.model = model
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
                 duration: float | np.ndarray,
                 return_period: float,
                 specific_discharge: float,
                 discharge_Q10: float):

        self.duration = np.asarray(duration)
        self.return_period = return_period
        self.specific_discharge = specific_discharge
        self.discharge_Q10 = discharge_Q10

        assert 0 <= duration.all()
        assert 0 <= return_period
        assert 0 <= specific_discharge
        assert 0 <= discharge_Q10

    def __matmul__(self, catchment):
        return qdf(catchment=catchment, rain=self)


class Event:

    def __init__(self, duration, discharge) -> None:

        self.duration = duration
        self.discharge = discharge

    def diagram(self, *args, **kwargs):
        return QDFdiagram(self, *args, **kwargs)


class QDFdiagram:

    def __init__(self,
                 event: Event,
                 style: str = "ggplot",
                 colors=("teal",
                         "k",
                         "indigo",
                         "tomato",
                         "green"),
                 flows_margin=0.3,
                 rain_margin=7,
                 show=True) -> None:

        self.event = event
        self.colors = colors
        self.flows_margin = flows_margin
        self.rain_margin = rain_margin

        duration = event.duration
        discharge = event.discharge

        with plt.style.context(style):

            c1, c2, c3, c4, c5 = self.colors
            fig, ax = plt.subplots()
            self.line, = ax.plot(duration, discharge, c=c1)

            ax.set_xlabel("Duration [h]")
            ax.set_ylabel("Discharge [m$^3$/s]")

            self.axes = (ax, )
            self.figure = fig
            if show:
                plt.show()

    def update(self, event: Event):

        self.line.set_data(event.duration, event.discharge)


def QDFapp(catchment: Catchment = None,
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
        ("rain", "discharge_Q10", "m3/s", "Q10")
    ]
    entries = map(lambda e: Entry(*e), entries)
    ModelApp(
        catchment=catchment,
        rain=rain,
        entries=entries
    )


def qdf(catchment, rain):

    constants = list(catchment._coefs[catchment.model].values())
    print(catchment.model)

    d = rain.duration
    if hasattr(catchment, "specific_duration"):
        ds = catchment.specific_duration
    else:
        ds = Turraza(
            catchment.surface,
            catchment.length,
            catchment.mean_slope
        )

    coefs = np.array([1/(a1*d/ds + a2) + a3
                      for a1, a2, a3 in constants])

    T = rain.return_period
    A, B, C = coefs

    if 0.5 <= T <= 20:
        discharge = rain.specific_discharge * (A * np.log(T) + B)
    elif T <= 1000:
        discharge = (
            rain.discharge_Q10 +
            rain.specific_discharge * C * np.log(1 + A * (T-10)/(10*C))
        )
    else:
        raise ValueError(
            f"{T = :.0f} is not within [0.5:1000] years"
        )

    event = Event(rain.duration, discharge)

    return event


def main():

    QDFapp(catchment=Catchment(model="soyans",
                               specific_duration=1,
                               surface=1.8,
                               length=2,
                               mean_slope=9.83/100))


if __name__ == "__main__":
    main()
