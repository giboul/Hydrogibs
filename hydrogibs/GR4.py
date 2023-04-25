import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
    NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import customtkinter as ctk
from typing import Callable
from dataclasses import dataclass


def _transfer_func(n: float, X4: float):  # m/km/s
    """
    This function will make the transition between the
    water flow and the discharge through a convolution

    discharge = convolution(_transfer_func(water_flow, time/X4))
    """
    if n < 1:
        return 3/(2*X4) * n**2
    if n < 2:
        return 3/(2*X4) * (2-n)**2
    return 0


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

    def __init__(self, time: np.ndarray, rain_func: Callable) -> None:

        self.time = time
        self.rain_func = rain_func
        self.rainfall = np.array([rain_func(t) for t in time])
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

        if observation_span is None:
            observation_span = 5*duration
        timestep = timestep if timestep is not None else duration/100
        time = np.arange(0, observation_span, timestep)

        rainfall = np.full_like(time, intensity)
        rainfall[time > duration] = 0

        self.time = time
        self.intensity = intensity
        self.duration = duration
        self.timestep = timestep if timestep is not None else duration/200
        self.observation_span = observation_span
        self.rainfall = rainfall

        assert 0 <= intensity
        assert 0 <= duration
        assert 0 <= timestep <= duration
        assert 0 <= observation_span > duration


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
                 transfer_function: Callable = _transfer_func) -> None:

        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4
        self.surface = surface
        self.transfer_function = transfer_function
        self.initial_volume = initial_volume

        assert 0 <= X1 <= 1, "Runoff coefficient must be within [0 : 1]"
        assert 0 <= X2, "Initial abstraction must be positive"
        assert 0 <= X3 <= 1, "Emptying rate must be within [0 : 1]"
        assert 0 <= X4, "Raising time must be positive"

    def __matmul__(self, rain):
        return rain @ self


@dataclass
class Event:

    time: np.ndarray
    rainfall: np.ndarray
    volume: np.ndarray
    water_flow_rain: np.ndarray
    water_flow_volume: np.ndarray
    water_flow: np.ndarray
    discharge_rain: np.ndarray
    discharge_volume: np.ndarray
    discharge: np.ndarray

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

        if isinstance(self.rain, BlockRain):
            self.apply = self.apply_block_rain
        else:
            self.apply = self.apply_rain

        self.apply()

    def apply_block_rain(self):

        self.event = gr4_block_rain(self.catchment, self.rain)

        return self.event

    def apply_rain(self):

        self.event = gr4_diff(self.catchment, self.rain)

        return self.event

    def create_diagram(self, *args, **kwargs):

        self.diagram = GR4diagram(self.event, *args, **kwargs)

    def App(self, *args, **kwargs):
        GR4App(self, show=False, *args, **kwargs)


def gr4_diff(catchment, rain):

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
    V = (
        np.exp(-X3*time) * (1-X1) *
        np.cumsum(np.exp(X3*time) * dP_effective) * dt
    )
    V = V + V0 * np.exp(-X3*time)

    t_abstraction = time < t1
    dTp = X1*dP
    dTp[t_abstraction] = 0
    dTv = X3*V
    dTv[t_abstraction] = 0

    q = np.array([
        catchment.transfer_function(ni, X4)
        for ni in time[time <= 2*X4]/X4
    ])

    Qp = S * np.convolve(dTp, q)[:time.size] * dt
    Qv = S * np.convolve(dTv, q)[:time.size] * dt

    return Event(time, dP, V, dTp, dTv, dTp+dTv, Qp, Qv, Qp+Qv)


def gr4_block_rain(catchment, block_rain) -> dict:
    """
    This method is fit for block events
    (constant rainfall intensity during a defined duration)

    Args:
        intensity         (float) [mm/h]: Ranfall intensity
        duration          (float) [h]: Rainfall duration
        observation_span  (float) [h]: Observation duration,
                                        default to 10*duration if not specified
        timestep          (float) [h]: Timestep,
                                        default to 0.01 if not specified
        inital_volume     (float) [mm]: Initial sub-surface volume,
                                        default to 0.0 if not specified
        transfer_function (callable): the transfer function,
                                        discharge = convolution(
                                            water_flow,
                                            transfer_function
                                        )

    Returns:
        GR4h object with the attributes:
            rainfall   (numpy 1Darray)
            volume     (numpy 1Darray)
            water_flow (numpy 1Darray)
            time       (numpy 1Darray)
            discharge  (numpy 1Darray)
    """

    # Unpack catchment attributes
    X1 = catchment.X1
    X2 = catchment.X2
    X3 = catchment.X3
    X4 = catchment.X4

    S = catchment.surface
    V0 = catchment.initial_volume

    transfer_function = catchment.transfer_function

    # Unpack rain attributes
    dt = block_rain.timestep
    t0 = block_rain.duration
    tf = block_rain.observation_span

    I0 = block_rain.intensity
    rainfall = block_rain.rainfall

    # Initializing time
    tf = 10*t0 if tf is None else tf
    t = np.arange(start=0, stop=tf, step=dt)

    # End of abstraction
    t1 = X2/I0

    # Initialize volume
    V = np.zeros_like(t, dtype=np.float32)
    V += V0

    dTp = np.zeros_like(t, dtype=np.float128)
    dTv = np.zeros_like(t, dtype=np.float128)

    # Initial abstraction
    i = t < t1
    A = np.zeros_like(t, dtype=np.float16)
    A[i] = I0*t[i]
    A[t >= t1] = A[i][-1]

    # Runoff + rain
    i = (t >= t1) & (t <= t0)
    V[i] += I0*(1-X1)/X3 * (1 - np.exp(-(t[i]-t1)*X3))
    V1 = V[i][-1] if V[i].size else 0

    dTp[i] = X1*I0
    dTv[i] = X3*V[i]

    # Runoff, no more rain
    i = t >= t0
    V[i] = V1 * np.exp(-(t[i]-t0)*X3)
    dTv[i] = X3*V[i]

    # Evaluate transfer function
    q = np.array([
        transfer_function(ni, X4)
        for ni in t[t <= 2*X4]/X4
    ])
    # Convolve
    Qp = S * np.convolve(dTp, q)[:t.size] * dt
    Qv = S * np.convolve(dTv, q)[:t.size] * dt

    return Event(t, rainfall, V, dTp, dTv, dTp+dTv, Qp, Qv, Qp+Qv)


class GR4App:

    def __init__(self, gr4: GR4h,
                 appearance: str = "dark",
                 color_theme: str = "dark-blue",
                 style: str = "seaborn",
                 close_and_clear: bool = True,
                 *args, **kwargs):

        self.gr4 = gr4

        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme(color_theme)

        self.root = ctk.CTk()
        self.root.title("Génie Rural 4")
        self.root.bind('<Return>', self.entries_update)
        # self.ww = self.root.winfo_screenwidth()
        # self.wh = self.root.winfo_screenheight()
        # self.root.geometry(f"{self.ww*0.8:.0f}x{self.wh*0.5:.0f}")

        self.dframe = ctk.CTkFrame(master=self.root)
        self.dframe.grid(row=0, column=1, sticky="NSEW")

        self.init_diagram(style=style, *args, **kwargs)

        self.pframe = ctk.CTkFrame(master=self.root)
        self.pframe.grid(column=0, row=0, sticky="NSEW")

        # entryframe = ctk.CTkLabel(text="GR4 parameters",
        #                           master=self.pframe,
        #                           font=("monospace", 24))
        # entryframe.grid(row=0, column=0,
        #                 sticky="NSEW",
        #                 ipadx=5, ipady=10)

        self.entries = dict()
        self.define_entry("X1", "-")
        self.define_entry("X2", "mm")
        self.define_entry("X3", "1/h")
        self.define_entry("X4", "h")
        self.define_entry("surface", "km²", "S")
        self.define_entry("initial_volume", "mm", "V0")

        if isinstance(self.gr4.rain, BlockRain):
            self.define_entry("observation_span", "mm", "tf")
            self.define_entry("intensity", "mm/h", "I0")
            self.define_entry("duration", "h", "t0")

        ctk.CTkButton(master=self.pframe,
                      text="Reset zoom",
                      command=lambda: self.diagram.zoom(self.canvas)
                      ).grid(pady=10)

        self.root.mainloop()
        if close_and_clear:
            plt.close()

    def init_diagram(self, *args, **kwargs):

        diagram = GR4diagram(self.gr4.event, *args, **kwargs)

        self.ax1, self.ax2, self.ax3 = diagram.axes

        self.canvas = FigureCanvasTkAgg(diagram.fig, master=self.dframe)
        toolbar = NavigationToolbar2Tk(self.canvas)
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

    def define_entry(self, key: str, unit, alias: str = None):

        entryframe = ctk.CTkFrame(master=self.pframe)
        entryframe.grid(sticky="NSEW")
        unit_str = f"[{unit}]"
        name = key if alias is None else alias
        label = ctk.CTkLabel(master=entryframe,
                             text=f"{name:>5} {unit_str:>6} ",
                             font=("monospace", 14))
        label.grid(row=0, column=0, sticky="EW", ipady=5)

        input = ctk.CTkEntry(master=entryframe)

        if hasattr(self.gr4.catchment, key):
            value = getattr(self.gr4.catchment, key)
            input.insert(0, value)
        elif hasattr(self.gr4.rain, key):
            value = getattr(self.gr4.rain, key)
            input.insert(0, value)
        else:
            raise KeyError(f"{key} not an attribute")

        input.grid(row=0, column=1, sticky="EW")

        slider = ctk.CTkSlider(master=entryframe,
                               from_=0, to=2*value if value else 1,
                               number_of_steps=999,
                               command=lambda _: self.slider_update(key))
        slider.grid(row=0, column=2, sticky="EW")

        self.entries[key] = dict(
            label=label,
            input=input,
            slider=slider
        )

    def slider_update(self, key):

        v = self.entries[key]["slider"].get()
        self.entries[key]["input"].delete(0, ctk.END)
        self.entries[key]["input"].insert(0, f"{v:.2f}")

        if hasattr(self.gr4.catchment, key):
            setattr(self.gr4.catchment, key, v)
        elif hasattr(self.gr4.rain, key):
            setattr(self.gr4.rain, key, v)
        else:
            raise KeyError(f"{key} not an attribute")

        if key == "observation_span":
            rain = self.gr4.rain
            self.gr4.rain = BlockRain(
                observation_span=v,
                intensity=rain.intensity,
                duration=rain.duration,
                timestep=rain.timestep
            )

        self.update()

    def entries_update(self, _):
        for key in self.entries:
            v = float(self.entries[key]['input'].get())
            self.entries[key]["slider"].configure(to=2*v if v else 1)
            if v:
                self.entries[key]["slider"].set(v)

            if key == "observation_span":
                rain = self.gr4.rain
                self.gr4.rain = BlockRain(
                    observation_span=v,
                    intensity=rain.intensity,
                    duration=rain.duration,
                    timestep=rain.timestep
                )
            elif key == "initial_volume":
                kwargs = self.gr4.catchment.__dict__
                del kwargs[key]
                self.gr4.catchment = Catchment(initial_volume=v, **kwargs)
            else:
                if hasattr(self.gr4.catchment, key):
                    setattr(self.gr4.catchment, key, v)
                elif hasattr(self.gr4.rain, key):
                    setattr(self.gr4.rain, key, v)
                else:
                    raise KeyError(f"{key} not an attribute")

        self.update()

    def update(self):

        self.gr4.apply()
        self.diagram.update(self.gr4.event, self.gr4.rain)
        self.canvas.draw()


def GR4_demo(kind="block"):

    if kind == "block":
        rain = BlockRain(50, duration=1.8)
    else:
        rain = Rain(
            time=np.linspace(0, 10, 1000),
            rain_func=lambda t: 50 if t < 2 else 0
        )
    GR4h(Catchment(8/100, 40, 0.1, 1), rain).App()


if __name__ == "__main__":
    GR4_demo()
