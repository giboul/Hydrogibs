import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,\
                                              NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from scipy.integrate import odeint
from scipy.optimize import OptimizeResult, minimize
from warnings import warn
import customtkinter as ctk


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


def transfer_func(n: float, X4: float):  # m/km/s
    """
    This function will make the transition between the
    water flow and the discharge through a convolution
    """
    if n < 1:
        return 3/(2*X4) * n**2
    if n < 2:
        return 3/(2*X4) * (2-n)**2
    return 0


class GR4h:
    """
    Implements the GR4h hydrological model.

    Parameters:
    X1 (float)  [-] : dQ = X1 * dPrecipitations
    X2 (float)  [mm]: Initial abstraction (vegetation interception)
    X3 (float) [1/h]: Sub-surface water volume emptying rate dSs = X3 * V*dt
    X4 (float)  [h] : the hydrogram's raising time
    """

    def __init__(self,
                 X1: float,
                 X2: float,
                 X3: float,
                 X4: float) -> None:
        """
        Parameters:
        X1 (float)  [-] : dQ = X1 * dPrecipitations
        X2 (float)  [mm]: Initial abstraction (vegetation interception)
        X3 (float) [1/h]: Sub-surface water volume emptying rate dSs = X3*V*dt
        X4 (float)  [h] : the hydrogram's raising time
        """
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.X4 = X4

    def block_rain(self, I0: float, t0: float,
                   S: float = 1.0, tf: float = None, dt: int = 0.01,
                   V0: float = 0.0, transfer_function: callable = None):
        """
        This method is fit for block events
        (constant rainfall intensity during a defined duration)

        Args:
            I0 (float) [mm/h]: Ranfall intensity
            t0 (float) [h]: Rainfall duration
            S  (float) [km^2]: Catchment surface,
                               default to 1 km^2 if not specified
            tf (float) [h]: Observation duration,
                            default to 10*t0 if not specified
            dt (float) [h]: Timestep,
                            default to 0.01 h if not specified
            V0 (float) [mm]: Initial sub-surface volume,
                             default to 0.0 if not specified
            transfer_function (callable): the transfer function,
                                          Q = convolution(
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

        # Unpack 4 parameters
        X1, X2, X3, X4 = self.X1, self.X2, self.X3, self.X4

        # Initializing time
        tf = 10*t0 if tf is None else tf
        t = np.arange(start=0, stop=tf, step=dt)

        # End of abstraction
        t1 = X2/I0

        # Initialize volume
        V = np.zeros_like(t, dtype=np.float32)
        V += V0

        dT = np.zeros_like(t, dtype=np.float128)

        # Initial abstraction
        i = t < t1
        A = np.zeros_like(t, dtype=np.float16)
        A[i] = I0*t[i]
        A[t >= t1] = A[i][-1]

        # Runoff + rain
        i = (t >= t1) & (t <= t0)
        V[i] += I0*(1-X1)/X3 * (1 - np.exp(-(t[i]-t1)*X3))
        V1 = V[i][-1]

        dT[i] = X1*I0 + X3*V[i]

        # Runoff, no more rain
        i = t >= t0
        V[i] = V1 * np.exp(-(t[i]-t0)*X3)
        dT[i] = X3*V[i]

        # Evaluate transfer function
        self.transferf = (transfer_func if transfer_function is None
                          else transfer_function)
        q = np.array([
            self.transferf(ni, X4)
            for ni in t[t <= 2*X4]/X4
        ])
        # Convolve
        alpha = 1  # 1/3.6
        Q = alpha * S * np.convolve(dT, q)*dt
        Q = Q[:t.size]
        tc = np.arange(start=0, stop=Q.size*dt, step=dt)

        # Save data to object
        self.time = t
        self.volume = V
        self.water_flow = dT
        self.discharge = Q
        self.rainfall = np.array([I0 if tau <= t0 else 0
                                 for tau in t])

        return self

    def rain(self, tf: float, rainfall_func: callable,
             S: float = 1.0, dt: float = 0.01,
             V0: float = 0.0, transfer_function: callable = None):
        """
        Calculates the discharge response to any rainfall event
        according to the GR4h method

        Args:
            I0 (float) [mm/h]: Ranfall intensity
            t0 (float) [h]: Rainfall duration
            S  (float) [km^2]: Catchment surface,
                               default to 1 km^2 if not specified
            tf (float) [h]: Observation duration,
                            default to 10*t0 if not specified
            rainfall_func (callable): rainfall function of time
            dt (float) [h]: Timestep, default to 0.01 h if not specified
            V0 (float) [mm]: Initial sub-surface volume,
                             default to 0.0 if not specified
            transfer_function (callable): discharge = convolution(
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

        self.tf = tf
        self.dt = dt

        self.S = S
        self.V0 = V0

        self.rainfall_func = rainfall_func
        self.transferf = (transfer_func if transfer_function is None
                          else transfer_function)

        return self.calculate()

    def calculate(self):

        X1, X2, X3, X4 = self.X1, self.X2, self.X3, self.X4

        time = np.arange(start=0, stop=self.tf, step=self.dt)
        dP = np.array([self.rainfall_func(t) for t in time])
        i = time[np.cumsum(dP)*self.dt >= X2]
        t1 = i[0] if i.size else float("inf")

        V = odeint(
            lambda V, t: (
                -X3*V + (1-X1)*self.rainfall_func(t) if t > t1 else -X3*V
            ),
            self.V0, time
        )[:, 0]

        t_abstraction = time < t1
        dTp = X1*dP
        dTp[t_abstraction] = 0
        dTv = X3*V
        dTv[t_abstraction] = 0
        dT = dTp + dTv

        q = np.array([
            self.transferf(ni, X4)
            for ni in time[time <= 2*X4]/X4
        ])

        self.time = time
        self.rainfall = dP
        self.volume = V
        self.water_flow = dT
        self.discharge_p = (self.S * np.convolve(dTp, q)*self.dt)[:time.size]
        self.discharge_v = (self.S * np.convolve(dTv, q)*self.dt)[:time.size]
        self.discharge = self.discharge_p + self.discharge_v

        return self

    def diagram(self, show=True, style: str = "ggplot",
                colors=("teal", "k", "indigo", "tomato", "green"), k=1.3):
        """Plots a diagram with rainfall, water flow and discharge"""

        with plt.style.context(style):

            c1, c2, c3, c4, c5 = colors

            fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
            ax1.set_title("Runoff response to rainfall")

            patch = ax1.fill_between(
                x=self.time,
                y1=self.discharge,
                y2=np.maximum(self.discharge_v, self.discharge_p),
                alpha=0.5,
                lw=0.0,
                color=c1,
                label="total discharge"
            )
            patch1 = ax1.fill_between(
                self.time,
                self.discharge_p,
                alpha=0.3,
                lw=0.0,
                color=c4,
                label="Runoff discharge"
            )
            patch2 = ax1.fill_between(
                self.time,
                self.discharge_v,
                alpha=0.3,
                lw=0.0,
                color=c5,
                label="Sub-surface discharge"
            )
            ax1.set_ylabel("$Q$ (m³/s)", color=c1)
            ax1.set_xlabel("Time [h]")
            ax1.set_xlim((self.time.min(), self.time.max()))
            ax1.set_ylim((0, k*self.discharge.max()))
            yticks = ax1.get_yticks()
            yticks = [y for y in yticks if y < max(yticks)/k]
            ax1.set_yticks(yticks)
            ax1.set_yticklabels(ax1.get_yticklabels(), color=c1)

            ax2 = ax1.twinx()
            bars = ax2.bar(
                self.time,
                self.rainfall,
                alpha=0.5,
                width=self.time[1]-self.time[0],
                color=c2,
                label="Rainfall"
            )
            max_rain = self.rainfall.max()
            ax2.set_ylim((8*max_rain, 0))
            ax2.grid(False)
            ax2.set_yticks((0, max_rain))
            ax2.set_yticklabels(ax2.get_yticklabels(), color=c2)

            ax3 = ax2.twinx()
            line, = ax3.plot(self.time, self.water_flow, "-.",
                             color=c3, label="Water flow", lw=1.5)
            ax3.set_ylabel("$\\dot{T}$ (mm/h)", color=c3)
            ax3.set_xlabel("$t$ (h)")
            ax3.set_ylim((0, k*self.water_flow.max()))
            yticks = ax3.get_yticks()
            yticks = [y for y in yticks if y < max(yticks)/k]
            ax3.set_yticks(yticks)
            ax3.set_yticklabels(ax3.get_yticklabels(), color=c3)
            ax3.grid(False)

            lines = (bars, patch, patch1, patch2, line)
            labs = [line.get_label() for line in lines]
            ax1.legend(lines, labs)

            plt.tight_layout()
            if show:
                plt.show()

        if not show:
            return fig, (ax1, ax2, ax3), lines

    def App(self):
        return GR4App(self)


class GR4App:

    def __init__(self, gr4: GR4h,
                 appearance: str = "dark",
                 color_theme: str = "dark-blue"):

        self.gr4 = gr4

        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme(color_theme)

        self.root = ctk.CTk()
        self.root.title("Génie Rural 4")
        self.root.bind('<Return>', self.entries_update)
        self.ww = self.root.winfo_screenwidth()
        self.wh = self.root.winfo_screenheight()
        self.root.geometry(f"{self.ww*0.8:.0f}x{self.wh*0.5:.0f}")

        self.dframe = ctk.CTkFrame(master=self.root)
        self.dframe.pack(side=ctk.RIGHT, fill="x", pady=0)
        self.draw_diagram()

        entryframe = ctk.CTkLabel(text="GR4 method", master=self.root)
        entryframe.pack(side=ctk.TOP)

        self.entries = dict()
        self.define_entry("X1", "-", self.gr4.X1)
        self.define_entry("X2", "mm", self.gr4.X2)
        self.define_entry("X3", "1/h", self.gr4.X3)
        self.define_entry("X4", "h", self.gr4.X4)
        self.root.mainloop()

    def define_entry(self, key: str, unit, value):

        entryframe = ctk.CTkFrame(master=self.root)
        entryframe.pack(side=ctk.TOP)
        unit_str = f"[{unit}]"
        label = ctk.CTkLabel(master=entryframe,
                             text=f"{key:>5} {unit_str:>5}",
                             font=("monospace", 12))
        label.pack(pady=10, padx=20, fill="x", side=ctk.LEFT)

        input = ctk.CTkEntry(master=entryframe)
        input.insert(0, value)
        input.pack(pady=10, padx=20, fill="both", side=ctk.LEFT)

        slider = ctk.CTkSlider(master=entryframe,
                               from_=0, to=2*value,
                               number_of_steps=999,
                               command=lambda _: self.slider_update(key))
        slider.pack(pady=10, padx=20, fill="x", side=ctk.RIGHT)

        self.entries[key] = dict(
            label=label,
            input=input,
            slider=slider
        )

    def slider_update(self, key):
        v = self.entries[key]["slider"].get()
        self.entries[key]["input"].delete(0, ctk.END)
        self.entries[key]["input"].insert(0, f"{v:.2f}")
        setattr(self.gr4, key, v)
        self.update_canvas()

    def entries_update(self, _):
        for key in self.entries:
            v = float(self.entries[key]['input'].get())
            self.entries[key]["slider"].configure(to=2*v)
            self.entries[key]["slider"].set(v)
            setattr(self.gr4, key, v)
        self.update_canvas()

    def draw_diagram(self):

        self.fig, axes, lines = self.gr4.diagram(show=False)
        self.ax1, self.ax2, self.ax3 = axes
        self.rain, self.disch, self.dischp, self.dischv, self.wflow = lines

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.dframe)
        toolbar = NavigationToolbar2Tk(self.canvas)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=ctk.TOP, fill="x")
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="x")
        self.canvas.mpl_connect('key_press_event',
                                lambda arg: key_press_handler(
                                    arg, self.canvas, toolbar
                                ))
        self.root.update()

    def update_canvas(self):
        self.gr4.calculate()
        t = self.gr4.time

        self.disch.set_verts((
            list(zip(  # transposing data
                np.concatenate((t, t[::-1])),
                np.concatenate((
                    self.gr4.discharge,
                    np.maximum(
                        self.gr4.discharge_p,
                        self.gr4.discharge_v)[::-1]
                ))
            )),
        ))
        self.dischp.set_verts((
            list(zip(t, self.gr4.discharge_p)) + [(t[-1], 0)],
        ))
        self.dischv.set_verts((
            list(zip(t, self.gr4.discharge_v)) + [(t[-1], 0)],
        ))
        self.wflow.set_data(t, self.gr4.water_flow)
        # for rect, h in zip(self.rain, self.gr4.rainfall):
        #     rect.set_height(h)

        self.canvas.draw()
        # self.root.update()


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


def GR4h_main():
    from matplotlib import pyplot as plt
    plt.style.use("bmh")

    S = 1  # 2.2  # [km^2]
    X1 = 8/100  # [-] dR = X1*dP  # TODO
    X2 = 40  # [mm] Interception par la végétation  # TODO
    X3 = 0.1  # [h^-1] dH = X3*V*dt, V = (1-X1)*I*dt  # TODO
    X4 = 1  # [h] temps de montée tm ≃ td  # TODO
    I0 = 50  # mm/h
    t0 = 2

    gr4 = GR4h(X1, X2, X3, X4)

    def rain_func(t):
        return I0*t if t < t0 else 0

    gr4 = gr4.rain(10, rain_func, S)  # .diagram(style="bmh")
    gr4.App()


def QdF_main():
    qdf = QdF(model="soyans", ds=1, S=1.8, L=2, im=25)
    Q10 = crupedix(S=1.8, Pj10=72, R=1.75)
    d = np.linspace(0, 3)
    Q100 = qdf.discharge(d, 100, Q10, Q10)
    plt.plot(d, Q100)
    plt.show()


if __name__ == "__main__":
    GR4h_main()
