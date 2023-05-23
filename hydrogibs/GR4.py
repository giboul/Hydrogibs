import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, Literal
from hydrogibs.ModelApp import ModelApp, Entry
import hydrogibs.ModelTemplate as ModelTemplate
from warnings import warn


def _transfer_func(X4: float, num: int) -> np.ndarray:
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


class Rain(ModelTemplate.Rain):
    """
    Rain object to apply to a Catchment object.

    Args:
        - time        (numpy.ndarray)  [h]
        - rainfall    (numpy.ndarray) [mm/h]

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
        if isinstance(self, BlockRain):
            return gr4(rain=self.to_rain(), catchment=catchment)
        return gr4(rain=self, catchment=catchment)


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
        return gr4(rain=self.to_rain(), catchment=catchment)


class Catchment(ModelTemplate.Catchment):
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


class Laval(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(57.6/100, 7.8, 2.4/100, 0.38,
                         *args, **kwargs)


class Erlenbach(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(46.5/100, 13.6, 16.2/100, 0.63,
                         *args, **kwargs)


class Rimbaud(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(35.4/100, 40, 2.28/100, 1.07,
                         *args, **kwargs)


class Latte(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(14.4/100, 75.4, 3.96/100, 0.78,
                         *args, **kwargs)


class Sapine(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(15.7/100, 71.1, 0.90/100, 1.03,
                         *args, **kwargs)


class Rietholzbach(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(26.5/100, 17, 2.82/100, 1.11,
                         *args, **kwargs)


class Lumpenenbach(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(22.6/100, 12.2, 9.6/100, 0.5,
                         *args, **kwargs)


class Vogelbach(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(31.4/100, 11.5, 5.88/100, 0.64,
                         *args, **kwargs)


class Brusquet(Catchment):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(13.8/100, 22.4, 0.72/100, 1.63,
                         *args, **kwargs)


presets = dict(
    Laval=Laval,
    Erlenbach=Erlenbach,
    Rimbaud=Rimbaud,
    Latte=Latte,
    Sapine=Sapine,
    Rietholzbach=Rietholzbach,
    Lumpenenbach=Lumpenenbach,
    Vogelbach=Vogelbach,
    Brusquet=Brusquet
)


class Event(ModelTemplate.Event):
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


class GR4diagram(ModelTemplate.Diagram):

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
                 figsize=(6, 3.5),
                 dpi=100,
                 show=True) -> None:

        self.colors = colors
        self.flows_margin = flows_margin
        self.rain_margin = rain_margin

        time = event.time
        rain = event.rainfall
        V = event.volume
        Qp = event.discharge_rain
        Qv = event.discharge_volume
        Q = event.discharge

        tmax = time.max()
        Qmax = Q.max()
        rmax = rain.max()
        Vmax = V.max()

        with plt.style.context(style):

            c1, c2, c3, c4, c5 = self.colors

            fig, (ax2, ax1) = plt.subplots(
                figsize=figsize,
                nrows=2, gridspec_kw=dict(
                    hspace=0,
                    height_ratios=[1, 3]
                ),
                dpi=dpi,
                sharex=True
            )
            ax2.invert_yaxis()
            ax2.xaxis.tick_top()
            ax3 = ax1.twinx()

            lineQ, = ax1.plot(
                time,
                Q,
                lw=2,
                color=c1,
                label="Débit",
                zorder=10
            )
            lineQp, = ax1.plot(
                time,
                Qp,
                lw=1,
                ls='-.',
                color=c4,
                label="Ruissellement",
                zorder=9
            )
            lineQv, = ax1.plot(
                time,
                Qv,
                lw=1,
                ls='-.',
                color=c5,
                label="Écoulements hypodermiques",
                zorder=9
            )
            ax1.set_ylabel("$Q$ (m³/s)", color=c1)
            ax1.set_xlabel("Temps (h)")
            ax1.set_xlim((0, tmax if tmax else 1))
            ax1.set_ylim((0, Qmax*1.1 if Qmax else 1))
            ax1.tick_params(colors=c1, axis='y')

            lineP, = ax2.step(
                time,
                rain,
                lw=1.5,
                color=c2,
                label="Précipitations"
            )
            ax2.set_ylim((rmax*1.2 if rmax else 1, -rmax/20))
            ax2.set_ylabel("$P$ (mm)")

            lineV, = ax3.plot(
                time,
                V,
                ":",
                color=c3,
                label="Volume de stockage",
                lw=1
            )
            ax3.set_ylim((0, Vmax*1.1 if Vmax else 1))
            ax3.set_ylabel("$V$ (mm)", color=c3)
            ax3.tick_params(colors=c3, axis='y')
            ax3.grid(False)

            ax1.spines[['top', 'right']].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax3.spines[['left', 'bottom', 'top']].set_visible(False)

            lines = (lineP, lineQ, lineQp, lineQv, lineV)
            labs = [line.get_label() for line in lines]
            ax3.legend(
                lines,
                labs,
                loc="upper right",
                frameon=True
            )

            plt.tight_layout()

            self.figure, self.axes, self.lines = fig, (ax1, ax2, ax3), lines

        if show:
            plt.show()

    def update(self, event):

        time = event.time
        rainfall = event.rainfall
        rain, discharge, discharge_p, discharge_v, storage_vol = self.lines

        discharge.set_data(time, event.discharge)
        discharge_p.set_data(time, event.discharge_rain)
        discharge_v.set_data(time, event.discharge_volume)
        storage_vol.set_data(time, event.volume)
        rain.set_data(time, rainfall)

    def zoom(self, canvas):

        rain, discharge, _, _, storage_vol = self.lines
        ax1, ax2, ax3 = self.axes

        t, Q = discharge.get_data()
        tmax = t.max()
        Qmax = Q.max()
        Imax = rain.get_data()[1].max()
        Vmax = storage_vol.get_data()[1].max()

        ax1.set_xlim((0, tmax if tmax else 1))
        ax1.set_ylim((0, Qmax*1.1 if Qmax else 1))
        ax2.set_ylim((Imax*1.2 if Imax else 1, -Imax/20))
        ax3.set_ylim((0, Vmax*1.1 if Vmax else 1))

        for ax in (ax1, ax2, ax3):
            ax.relim()

        plt.tight_layout()
        canvas.draw()


def App(catchment: Catchment = None,
        rain: Rain = None,
        *args, **kwargs):
    if catchment is None:
        catchment = Catchment(8/100, 40, 0.1, 1)
    if rain is None:
        rain = BlockRain(50, duration=1.8)
    entries = [
        ("catchment", "X1", "-"),
        ("catchment", "X2", "mm"),
        ("catchment", "X3", "1/h"),
        ("catchment", "X4", "h"),
        ("catchment", "surface", "km²", "S"),
        ("catchment", "initial_volume", "mm", "V0"),
    ]

    if isinstance(rain, BlockRain):
        entries += [
            ("rain", "observation_span", "mm", "tf"),
            ("rain", "intensity", "mm/h", "I0"),
            ("rain", "duration", "h", "t0")
        ]
    entries = map(lambda e: Entry(*e), entries)
    ModelApp(
        catchment=catchment,
        rain=rain,
        title="Génie rural 4",
        entries=entries,
        *args,
        **kwargs
    )


def gr4(catchment, rain, volume_check=False):

    # Unpack GR4 parameters
    X1 = catchment.X1  # [-]
    X2 = catchment.X2  # mm
    X3 = catchment.X3  # 1/h
    X4 = catchment.X4  # h

    # Other conditions
    S = catchment.surface  # km²
    V0 = catchment.initial_volume  # mm

    # Rainfall data
    time = rain.time  # h
    dt = rain.timestep  # h
    dP = rain.rainfall  # mm/h

    # integral(rainfall)dt >= initial abstraction
    abstraction = np.cumsum(dP)*dt < X2

    # Removing the initial abstraction from the rainfall
    dP_effective = dP.copy()
    dP_effective[abstraction] = 0

    # solution to the differential equation V' = -X3*V + (1-X1)*P
    V = np.exp(-X3*time) * (
        # homogeneous solution
        (1-X1) * np.cumsum(np.exp(X3*time) * dP_effective) * dt
        # particular solution / initial condition
        + V0
    )

    # Water flows
    dTp = X1*dP  # due to runoff
    dTv = X3*V  # due to volume emptying
    dTp[abstraction] = 0
    dTv[abstraction] = 0

    # transfer function as array
    q = catchment.transfer_function(X4, num=(time <= 2*X4).sum())

    Qp = S * np.convolve(dTp, q)[:time.size] * dt / 3.6
    Qv = S * np.convolve(dTv, q)[:time.size] * dt / 3.6

    Vtot = np.trapz(x=time, y=Qp + Qv)*3600
    Ptot = np.trapz(x=time, y=dP)*S*1000
    X2v = X2*S*1000 if (~abstraction).any() else Ptot
    if volume_check:
        warn(
            "\n"
            f"Stored volume: {Vtot + X2v:.2e}\n"
            f"\tDischarge     volume: {Vtot:.2e}\n"
            f"\tInitial  abstraction: {X2v:.2e}\n"
            f"Precipitation volume: {Ptot:.2e}\n\n"
        )

    return Event(time, dP, V, dTp+dTv, Qp, Qv, Qp+Qv)


def GR4_demo(kind: Literal["array", "block"] = "array"):

    if kind == "block":
        rain = BlockRain(50, duration=1.8, observation_span=5.8)
    else:
        time = np.linspace(0, 10, 1000)
        rainfall = np.full_like(time, 50)
        rainfall[(3 <= time) & (time <= 7) | (time >= 9)] = 0
        rain = Rain(
            time=time,
            rainfall=rainfall
        )
    Catchment(8/100, 40, 0.1, 1) @ rain
    # GR4h(Catchment(8/100, 40, 0.1, 1), rain).App()
    App(Catchment(8/100, 40, 0.1, 1), rain)


if __name__ == "__main__":
    GR4_demo("block")
