import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Literal
from ModelApp import ModelApp, Entry


def _transfer_func(X4: float, num: int) -> np.ndarray:  # m/km/s
    """
    This function will make the transition between the
    water flow and the discharge through a convolution

    discharge = convolution(_transfer_func(water_flow, time/X4))

    Args:
        - X4  (float): the hydrogram's raising time
        - num  (int) : the number of elements to give to the array

    Returns:
        - f (np.ndarray): = 3/(2*X4) * n**2            if n <= 1
                            3/(2*X4) * (2-n[n > 1])**2 if n >  1
    """
    n = np.linspace(0, 2, num)
    f = 3/(2*X4) * n**2
    f[n > 1] = 3/(2*X4) * (2-n[n > 1])**2
    return f


class Rain:
    """
    Rain object to apply to a Catchment object.

    Args:
        - time        (np.ndarray)       [h]
        - rain_func   (callable)   -> [mm/h]

    Creates a GR4h object when called with a Catchment object:
    >>> gr4h = GR4h(catchment, rain)
    Creates an Event object when applied to a catchment
    >>> event = rain @ catchment
    """

    def __init__(self, time: np.ndarray, rainfall: np.ndarray) -> None:

        self.time = np.asarray(time)
        self.rainfall = np.asarray(rainfall)
        self.timestep = time[1] - time[0]

    def __matmul__(self, catchment):
        return GR4h(catchment, self).apply()


class BlockRain(Rain):
    """
    A constant rain with a limited duration.

    Args:
        - intensity        (floaat)[mm/h]
        - duration         (float) [h]
        - timestep         (float) [h]: directly linked to precision
        - observation_span (float) [h]: the duration of the experiment

    Creates a GR4h object when called with a Catchment object:
    >>> gr4h = GR4h(catchment, rain)
    Creates an Event object when applied to a catchment
    >>> event = rain @ catchment
    """

    def __init__(self,
                 intensity: float,
                 duration: float = 1.0,
                 timestep: float = None,
                 observation_span: float = None) -> None:

        timestep = timestep if timestep is not None else duration/200
        observation_span = (observation_span if observation_span
                            else 5 * duration)

        assert 0 <= intensity
        assert 0 <= duration
        assert 0 <= timestep <= duration
        assert 0 <= observation_span > duration

        self.intensity = intensity
        self.duration = duration
        self.timestep = timestep
        self.observation_span = observation_span

    def to_rain(self):

        time = np.arange(0, self.observation_span, self.timestep)
        rainfall = np.full_like(time, self.intensity)
        rainfall[time > self.duration] = 0

        self.time = time
        self.rainfall = rainfall

        return self

    def __matmul__(self, catchment):
        return self.to_rain() @ catchment


class Catchment:
    """
    Stores GR4h catchment parameters.

    Creates a GR4h object when called with a Rain object:
    >>> gr4h = GR4h(catchment, rain)
    Creates an Event object when applied to a Rain object
    >>> event = rain @ catchment

    Args:
        X1 (float)  [-] : dQ = X1 * dPrecipitations
        X2 (float)  [mm]: Initial abstraction (vegetation interception)
        X3 (float) [1/h]: Sub-surface water volume emptying rate dQs = X3*V*dt
        X4 (float)  [h] : the hydrogram's raising time
    """

    def __init__(self,
                 X1: float,
                 X2: float,
                 X3: float,
                 X4: float,
                 surface: float = 1,
                 initial_volume: float = 0,
                 transfer_function: Callable = None) -> None:

        assert 0 <= X1 <= 1, "Runoff coefficient must be within [0 : 1]"
        assert 0 <= X2, "Initial abstraction must be positive"
        assert 0 <= X3 <= 1, "Emptying rate must be within [0 : 1]"
        assert 0 <= X4, "Raising time must be positive"

        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.surface = surface
        self.transfer_function = (transfer_function
                                  if transfer_function is not None
                                  else _transfer_func)
        self.initial_volume = initial_volume

    def __matmul__(self, rain):
        return rain @ self


class Event:
    """
    Stores all relevant results of a GR4h calculation

    basic class instead of dataclass, namedtuple or dataframe is used
    for speed reasons (an event will be created at every diagram update)
    """

    def __init__(self,
                 time: np.ndarray,
                 rainfall: np.ndarray,
                 volume: np.ndarray,
                 water_flow: np.ndarray,
                 discharge_rain: np.ndarray,
                 discharge_volume: np.ndarray,
                 discharge: np.ndarray) -> None:

        self.time = time
        self.rainfall = rainfall
        self.volume = volume
        self.water_flow = water_flow
        self.discharge_rain = discharge_rain
        self.discharge_volume = discharge_volume
        self.discharge = discharge

    def diagram(self, *args, **kwargs):
        return GR4diagram(self, *args, **kwargs)


class GR4diagram:

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

        self.colors = colors
        self.flows_margin = flows_margin
        self.rain_margin = rain_margin

        self.draw(event, style=style, show=show)

    def draw(self, event: Event, style: str = "seaborn", show=True):
        """Plots a diagram with rainfall, water flow and discharge"""

        time = event.time
        rain = event.rainfall
        dT = event.water_flow
        Qp = event.discharge_rain
        Qv = event.discharge_volume
        Q = event.discharge

        with plt.style.context(style):

            c1, c2, c3, c4, c5 = self.colors

            fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=100)
            ax1.set_title("Runoff response to rainfall")

            patch = ax1.fill_between(
                x=time,
                y1=Q,
                y2=np.maximum(Qv, Qp),
                alpha=0.5,
                lw=0.0,
                color=c1,
                label="total discharge"
            )
            patch1 = ax1.fill_between(
                time,
                Qp,
                alpha=0.3,
                lw=0.0,
                color=c4,
                label="Runoff discharge"
            )
            patch2 = ax1.fill_between(
                time,
                Qv,
                alpha=0.3,
                lw=0.0,
                color=c5,
                label="Sub-surface discharge"
            )
            ax1.set_ylabel("$Q$ (m³/s)", color=c1)
            ax1.set_xlabel("Time [h]")
            ax1.set_xlim((time.min(), time.max()))
            ax1.set_ylim((0, (1 + self.flows_margin)*Q.max()))
            ax1.set_yscale("linear")
            yticks = ax1.get_yticks()
            yticks = [
                y for y in yticks
                if y < max(yticks)/(self.flows_margin + 1)
            ]
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(yticks, color=c1)

            ax2 = ax1.twinx()
            bars = ax2.bar(
                time,
                rain,
                alpha=0.5,
                width=time[1]-time[0],
                color=c2,
                label="Rainfall"
            )
            max_rain = rain.max()
            ax2.set_ylim(((1 + self.rain_margin) * max_rain, 0))
            ax2.grid(False)
            ax2.set_yticks((0, max_rain))
            ax2.set_yticklabels(ax2.get_yticklabels(), color=c2)

            ax3 = ax2.twinx()
            line, = ax3.plot(time, dT, "-.",
                             color=c3, label="Water flow", lw=1.5)
            ax3.set_ylabel("$\\dot{T}$ (mm/h)", color=c3)
            ax3.set_xlabel("$t$ (h)")
            ax3.set_ylim((0, (1 + self.flows_margin) * dT.max()))
            yticks = ax3.get_yticks()
            yticks = [
                y for y in yticks
                if y < max(yticks)/(1 + self.flows_margin)
            ]
            ax3.set_yticks(yticks)
            ax3.set_yticklabels(ax3.get_yticks(), color=c3)
            ax3.set_yscale("linear")
            ax3.grid(False)

            lines = (bars, patch, patch1, patch2, line)
            labs = [line.get_label() for line in lines]
            ax1.legend(lines, labs)

            plt.tight_layout()

            self.fig, self.axes, self.lines = fig, (ax1, ax2, ax3), lines

        if show:
            plt.show()
        return self

    def update(self, event, rain_obj):

        t = event.time
        rain, discharge, discharge_p, discharge_v, water_flow = self.lines

        discharge.set_verts((
            list(zip(  # transposing data
                np.concatenate((t, t[::-1])),
                np.concatenate((
                    event.discharge,
                    np.maximum(
                        event.discharge_rain,
                        event.discharge_volume)[::-1]
                ))
            )),
        ))
        discharge_p.set_verts((
            list(zip(t, event.discharge_rain)) + [(t[-1], 0)],
        ))
        discharge_v.set_verts((
            list(zip(t, event.discharge_volume)) + [(t[-1], 0)],
        ))
        water_flow.set_data(t, event.water_flow)

        if isinstance(rain_obj, BlockRain):
            I0 = rain_obj.intensity
            d = rain_obj.duration
            for rect, v in zip(rain, t):
                if v <= d:
                    rect.set_height(I0)
                else:
                    rect.set_height(0)

    def zoom(self, canvas):

        rain, discharge, _, _, water_flow = self.lines
        ax1, ax2, ax3 = self.axes

        t, Q = discharge.get_paths()[0].vertices.T
        Qm = Q.max()
        Imax = max([b.get_height() for b in rain])
        _, dT = water_flow.get_data()
        dTm = dT.max()

        ax1.set_yscale("linear")
        ylim = Qm * (1 + self.flows_margin)
        ax1.set_ylim((0, ylim if ylim else 1))
        ax1.set_xlim((0, t.max()))
        yticks = [
            ytick for ytick in ax1.get_yticks()
            if ytick <= Qm
        ]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(yticks)

        ax2.set_yscale("linear")
        ylim = Imax * (1 + self.rain_margin)
        ax2.set_ylim((ylim if ylim else 1, 0))
        ax2.set_yticks((0, Imax))

        ax3.set_yscale("linear")
        ylim = dTm * (1 + self.flows_margin)
        ax3.set_ylim((0, ylim if ylim else 1))

        plt.tight_layout()
        canvas.draw()


class GR4h:
    """
    Object storing a Catchment object, a Rain object, and Event object
    and eventually attributes relative to a diagram

    A GR4h object is obtained when called with a Rain and a Catchment objects:
        >>> catchment = Catchment(X1=8/100, X2=40, X3=0.1, X4=1)
        >>> rain = BlockRain(intensity=50)
        >>> gr4h: GR4h = GR4h(catchment, rain)  # second syntax
        >>> gr4h.App()  # opens an interactive diagram in a tkinter window

    Args:
        catchment (Catchment): contains essential parameters
        rain      (Rain): contains the rainfall event details

    Returns:
        gr4h (GR4h): Object contaning an Event object (discharges, water flow)
        gr4h.event (Event): Contains the following arrays:
                                - volume
                                - water_flow
                                - discharge
                            The corresponding time is stored in gr4h.rain.time
    """

    def __init__(self, catchment: Catchment, rain: Rain) -> None:

        self.catchment = catchment
        self.rain = rain

        self.apply()

    def apply(self):

        rain = self.rain
        if isinstance(rain, BlockRain):
            rain = rain.to_rain()

        self.event = gr4(self.catchment, rain)

        return self.event

    def diagram(self, *args, **kwargs):

        self.diagram = GR4diagram(self.event, *args, **kwargs)

        return self.diagram

    def App(self, *args, **kwargs):
        entries = [
            ("catchment", "X1", "-"),
            ("catchment", "X2", "mm"),
            ("catchment", "X3", "1/h"),
            ("catchment", "X4", "h"),
            ("catchment", "surface", "km²", "S"),
            ("catchment", "initial_volume", "mm", "V0"),
        ]

        if isinstance(self.rain, BlockRain):
            entries += [
                ("rain", "observation_span", "mm", "tf"),
                ("rain", "intensity", "mm/h", "I0"),
                ("rain", "duration", "h", "t0")
            ]
        entries = [Entry(*entry) for entry in entries]
        ModelApp(self, title="Génie rural 4", entries=entries, *args, **kwargs)


def gr4(catchment, rain):

    X1 = catchment.X1
    X2 = catchment.X2
    X3 = catchment.X3
    X4 = catchment.X4
    S = catchment.surface
    V0 = catchment.initial_volume

    time = rain.time
    dt = rain.timestep
    dP = rain.rainfall

    i = time[np.cumsum(dP)*dt >= X2 - V0]
    t1 = i[0] if i.size else float("inf")

    dP_effective = dP.copy()
    dP_effective[time < t1] = 0

    # solution to the differential equation V' = -X3*V + (1-X1)*P
    integral = np.cumsum(np.exp(X3*time) * dP_effective) * dt
    cond_init = V0 * np.exp(-X3*time)
    V = np.exp(-X3*time) * (1-X1) * integral + cond_init

    t_abstraction = time < t1
    dTp = X1*dP
    dTv = X3*V
    dTp[t_abstraction] = 0
    dTv[t_abstraction] = 0

    q = catchment.transfer_function(X4, num=(time <= 2*X4).sum())

    Qp = S * np.convolve(dTp, q)[:time.size] * dt
    Qv = S * np.convolve(dTv, q)[:time.size] * dt

    return Event(time, dP, V, dTp+dTv, Qp, Qv, Qp+Qv)


def GR4_demo(kind: Literal["array", "block"] = "array"):

    if kind == "block":
        rain = BlockRain(50, duration=1.8)
    else:
        time = np.linspace(0, 10, 1000)
        rainfall = np.full_like(time, 50)
        rainfall[(3 <= time) & (time <= 7) | (time >= 9)] = 0
        rain = Rain(
            time=time,
            rainfall=rainfall
        )
    GR4h(Catchment(8/100, 40, 0.1, 1), rain).App()


if __name__ == "__main__":
    GR4_demo("block")
