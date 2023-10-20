import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from dataclasses import dataclass
from warnings import warn
from datetime import datetime
from matplotlib.dates import DateFormatter
from os.path import join, dirname


r"""
This module is fully dedicated to the GR4h method

It contains:
    - a Catchment object, storing the relevant catchment parameters
    - a Rain object, storing the relevant rain event data
    - a BlockRain object, simplifying the use of Rain objects
    - an Event object, storing the results of the GR4j(catchment, rain) model
    - a Diagram object for the quick representation of a hyetograph
    - an App to assess the effects of the catchment and rain parameters
"""


@dataclass(slots=True)
class Rain:
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

    time: np.ndarray
    rainfall: np.ndarray
    _seconds: np.ndarray = None

    def __post_init__(self):

        t0 = self.time[0]
        if isinstance(t0, datetime):
            self._seconds = np.array([
                d.total_seconds() for d in self.time - t0
            ]) / 3600

    @property
    def duration(self):
        return self._seconds if self._seconds is not None else self.time

    def __matmul__(self, catchment):
        return gr4(rain=self, catchment=catchment)


@dataclass(slots=True)
class BlockRain:
    r"""
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

    intensity: float
    duration: float
    timestep: float = None
    observation_span: float = None

    def __post_init__(self) -> None:

        if self.observation_span is None:
            self.observation_span = 5 * self.duration
        if self.timestep is None:
            self.timestep = self.duration / 200

    def to_rain(self):
        t = np.arange(0, self.observation_span, step=self.timestep)
        r = np.full_like(t, self.intensity)
        r[t > self.duration] = 0
        return Rain(t, r)

    def __matmul__(self, catchment):
        return self.to_rain() @ catchment


def _transfer_func(X4: float, num: int) -> np.ndarray:
    """
    This function will make the transition between the
    water flow and the discharge through a convolution

    discharge = convolution(_transfer_func(water_flow, time/X4))

    Args:
        - X4  (float): the hydrograph's raising time
        - num  (int) : the number of elements to give to the array

    Returns:
        - f (np.ndarray): = 3/(2*X4) * n**2            if n <= 1
                            3/(2*X4) * (2-n[n > 1])**2 if n >  1
    """
    n = np.linspace(0, 2, num)
    f = 3/(2*X4) * n**2
    f[n > 1] = 3/(2*X4) * (2-n[n > 1])**2
    return f


@dataclass(slots=True)
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
        X4 (float)  [h] : the hydrograph's raising time
    """

    X1: float
    X2: float
    X3: float
    X4: float
    surface: float = 1
    initial_volume: float = 0
    transfer_function: Callable = _transfer_func

    @property
    def X(self):
        return self.X1, self.X2, self.X3, self.X4

    def __matmul__(self, rain):
        return rain @ self


@dataclass(slots=True)
class Event:
    """
    Stores relevant results of a GR4h calculation
    """

    time: np.ndarray
    rainfall: np.ndarray
    volume: np.ndarray
    water_flow: np.ndarray
    discharge_rain: np.ndarray
    discharge_volume: np.ndarray
    discharge: np.ndarray

    def hydrograph(self, *args, **kwargs):
        return Diagram(self, *args, **kwargs)


def gr4(catchment: Catchment, rain: Rain, volume_check=False) -> Event:
    """
    This function computes an flood Event
    based on the given Catchment and Rain event
    """

    # Unpack GR4 parameters
    X1, X2, X3, X4 = catchment.X

    # Other conditions
    S = catchment.surface  # km²
    V0 = catchment.initial_volume  # mm

    # Rainfall data
    time = rain.duration  # h
    dP = rain.rainfall  # mm/h
    dt = np.diff(time, append=2*time[-1]-time[-2])  # h

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
    dTp = X1*dP_effective  # due to runoff
    dTv = X3*V  # due to volume emptying

    # transfer function as array
    q = catchment.transfer_function(X4, num=(time <= 2*X4).sum())

    Qp = S * np.convolve(dTp, q)[:time.size] * dt / 3.6
    Qv = S * np.convolve(dTv, q)[:time.size] * dt / 3.6

    Vtot = np.trapz(x=time, y=Qp + Qv)*3600
    Ptot = np.trapz(x=time, y=dP)*S*1000
    X2v = X2*S*1000 if (~abstraction).any() else Ptot
    if volume_check:
        print(
            "\n"
            f"Stored volume: {Vtot + X2v:.2e}\n"
            f"\tDischarge     volume: {Vtot:.4e}\n"
            f"\tInitial  abstraction: {X2v:.2e}\n"
            f"Precipitation volume: {Ptot:.2e}"
        )

    return Event(rain.time, dP, V, dTp+dTv, Qp, Qv, Qp+Qv)


with open(join(dirname(__file__), 'GR4.csv')) as file:
    """
    Creating the presets such that:
    >>> GR4presetsPresets[preset] = (X1, X2, X3)
    """
    GR4CatchmentPresets = {
        preset: (float(x1)/100, float(x2), float(x3)/100, float(x4))
        for preset, _surface, _region, x1, x2, x3, x4, _RTratio, _group
        in [
            line.split(',')  # 'cause .csv
            for line in file.read().splitlines()[1:]  # remove header
        ]
    }


class PresetCatchment(Catchment):

    def __init__(self, model: str, *args, **kwargs) -> None:

        model = model.capitalize()
        super().__init__(*GR4CatchmentPresets[model], *args, **kwargs)
        self.model = model


class Diagram:

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

        tmin = time.min()
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
            ax1.set_xlabel("t (h)")
            ax1.set_xlim((tmin, tmax if tmax else 1))
            ax1.set_ylim((0, Qmax*1.1 if Qmax else 1))
            ax1.tick_params(colors=c1, axis='y')
            if isinstance(tmin, datetime):
                span = (tmax-tmin).total_seconds()
                dateformat = None
                if span < 3600 * 24 * 31:
                    dateformat = "%b-%d %H:%M"
                if span < 3600 * 24:
                    dateformat = "%H:%M"
                elif span < 3600:
                    dateformat = "%M"
                if dateformat is not None:
                    ax1.xaxis.set_major_formatter(DateFormatter(dateformat))
                ax1.set_xticks(ax1.get_xticks())
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
                ax1.set_xlabel("t (dates)")

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

    def home_zoom(self, canvas):

        rain, discharge, _, _, storage_vol = self.lines
        ax1, ax2, ax3 = self.axes

        t, Q = discharge.get_data()
        tmin = t.min()
        tmax = t.max()
        Qmax = Q.max()
        Imax = rain.get_data()[1].max()
        Vmax = storage_vol.get_data()[1].max()

        ax1.set_xlim((tmin, tmax if tmax else 1))
        ax1.set_ylim((0, Qmax*1.1 if Qmax else 1))
        ax2.set_ylim((Imax*1.2 if Imax else 1, -Imax/20))
        ax3.set_ylim((0, Vmax*1.1 if Vmax else 1))

        for ax in (ax1, ax2, ax3):
            ax.relim()

        plt.tight_layout()
        canvas.draw()


class App:

    def __init__(self,
                 catchment: Catchment,
                 rain: Rain,
                 title: str = None,
                 appearance: str = "light",
                 color_theme: str = "blue",
                 style: str = "seaborn",
                 close_and_clear: bool = True,
                 *args, **kwargs):

        try:
            import customtkinter as ctk
        except ImportError:
            raise ("Install customtkinter for interactive apps")

        self.catchment = catchment
        self.rain = rain
        self.event = rain @ catchment

        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme(color_theme)

        self.root = ctk.CTk()
        self.root.title(title)
        self.root.bind('<Return>', self.entries_update)

        self.dframe = ctk.CTkFrame(master=self.root)
        self.dframe.grid(row=0, column=1, sticky="NSEW")

        self.init_diagram(style=style, show=False, *args, **kwargs)

        self.pframe = ctk.CTkFrame(master=self.root)
        self.pframe.grid(column=0, row=0, sticky="NSEW")

        entries = [
            ("catchment", "X1", "-"),
            ("catchment", "X2", "mm"),
            ("catchment", "X3", "1/h"),
            ("catchment", "X4", "h"),
            ("catchment", "surface", "km²", "S"),
            ("catchment", "initial_volume", "mm", "V0"),
        ]
        entries += [
            ("rain", "observation_span", "mm", "tf"),
            ("rain", "intensity", "mm/h", "I0"),
            ("rain", "duration", "h", "t0")
        ] if isinstance(rain, BlockRain) else []

        self.entries = dict()
        for row, entry in enumerate(entries, start=1):

            object, key, unit, *alias = entry

            entryframe = ctk.CTkFrame(master=self.pframe)
            entryframe.grid(sticky="NSEW")
            unit_str = f"[{unit}]"
            name = alias[0] if alias else key

            label = ctk.CTkLabel(
                master=entryframe,
                text=f" {name:<5} {unit_str:<6} ",
                font=("monospace", 14)
            )
            label.grid(row=row, column=0, sticky="EW", ipady=5)

            input = ctk.CTkEntry(master=entryframe, width=50)

            value = getattr(getattr(self, object), key)
            input.insert(0, value)
            input.grid(row=row, column=1, sticky="EW")

            slider = ctk.CTkSlider(
                master=entryframe,
                from_=0, to=2*value if value else 1,
                number_of_steps=999,
                command=(
                    lambda _, object=object, key=key:
                    self.slider_update(object, key)
                )
            )
            slider.grid(row=row, column=2, sticky="EW")

            self.entries[key] = dict(
                object=object,
                label=label,
                input=input,
                slider=slider
            )

        ctk.CTkButton(master=self.pframe,
                      text="Reset zoom",
                      command=lambda: self.diagram.home_zoom(self.canvas)
                      ).grid(pady=10)

        self.root.mainloop()
        if close_and_clear:
            plt.close()

    def init_diagram(self, *args, **kwargs):

        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk
        )
        from matplotlib.backend_bases import key_press_handler

        diagram = self.event.hydrograph(*args, **kwargs)

        self.canvas = FigureCanvasTkAgg(diagram.figure, master=self.dframe)
        toolbar = NavigationToolbar2Tk(
            canvas=self.canvas, window=self.dframe)
        toolbar.update()
        self.canvas._tkcanvas.pack()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect('key_press_event',
                                lambda arg: key_press_handler(
                                    arg, self.canvas, toolbar
                                ))
        self.diagram = diagram
        self.root.update()

    def slider_update(self, object: str, key: str):

        value = self.entries[key]["slider"].get()
        self.entries[key]["input"].delete(0, 'end')
        self.entries[key]["input"].insert(0, f"{value:.2f}")
        setattr(getattr(self, object), key, value)
        self.update()

    def entries_update(self, _KeyPressEvent):

        for key in self.entries:

            entry = self.entries[key]
            value = float(entry["input"].get())
            setattr(getattr(self, entry["object"]), key, value)
            v = value if value else 1
            slider = entry["slider"]
            slider.configure(to=2*v)
            slider.set(v)

        self.update()

    def update(self):

        event = self.rain @ self.catchment
        self.diagram.update(event)
        self.canvas.draw()


def test(plot=False, app=False, datetime=False):
    # Custom test
    X1 = 57.6/100  # [-] dR = X1*dP
    X2 = 7.28  # [mm] Interception par la végétation
    X3 = 2.4/100  # [h^-1] dH = X3*V*dt, V = (1-X1)*I*dt
    X4 = 0.38  # [h] temps de montée tm ≃ td

    t0 = 1  # h
    I0 = 66.7  # mm/h

    # df = pd.read_csv(join(dirname(__file__), "rain.csv"))
    # time = pd.to_datetime(df.Date, format="%Y-%m-%d %H:%M:%S")
    # rainfall = df.Rainfall
    dt = 0.01
    if datetime is False:
        time = np.arange(0, 24, step=dt)
        rainfall = np.exp(-(time - 3)**2)
        rainfall = I0 * rainfall / np.trapz(x=time, y=rainfall)
    else:
        import pandas as pd

        df = pd.read_csv("hydrogibs/test/floods/rain.csv")
        rainfall = df.Rainfall
        time = df.Date

    rain = Rain(time, rainfall)
    catchment = Catchment(X1, X2, X3, X4, surface=1.8)

    event = rain @ catchment

    if plot:
        Qax, Pax, Vax = event.diagram(show=False).axes
        Pax.set_title("Rimbaud")
        plt.show()

    # rain = GR4.BlockRain(I0, t0)
    catchment = PresetCatchment('Laval')
    event = rain @ catchment

    if app:
        App(catchment, rain)


if __name__ == "__main__":
    test(app=True)
