from scipy.interpolate import interp1d
from typing import Iterable
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
if __name__ == "__main__":
    from hydrogibs import constants as cst
else:
    from .. import constants as cst

rho_s = 2650
rho = cst.rho
g = cst.g
nu_k = cst.nu_k


def adimensional_diameter(diameter: float, solid_density: float, nu=nu_k):
    """
    Compute the adimensional diameter.

    d* = d * ((rho_s/rho_w - 1) * g/nu_k^2)^(1/3)

    Parameters
    ----------
    diameter : float
    solid_density : float
    nu_k : float = 1.316e-06
        The kinematic viscosity of water
    
    Returns
    -------
    float
        The adimensional diameter
    """
    return diameter*((solid_density/rho-1)*g/nu_k**2)**(1/3)


def reynolds(u_star: float, diameter: float, nu=nu_k):
    """
    Compute the Reynolds number of the solid grains.

    R = u* * d / nu_k

    Parameters
    ----------
    u_star : float
        The adimensional speed
    diameter : float
    nu_k : float = 1.316e-06
        The kinematic viscosity of water
    
    Returns
    -------
    float
        The Reynolds number
    """
    return u_star * diameter / nu


def adimensional_shear(shear: float, d: float, solid_density: float, g=g):
    """
    Compute the adimensional shear stress on the riverbed.

    t = shear / ((rho_s - rho) * g * d)

    Parameters
    ----------
    shear : float
        The shear stress on the riverbed
    d : float
        The grain diameter
    solid_density : float
        The density of the grains
    g : float = 9.81
        gravity
    
    Returns
    -------
    float
        Adimensional shear
    """
    return shear/((solid_density - rho)*g*d)


def smart_jaeggi(h: float, i: float, s: float, theta_cr: float, Dm: float):
    """
    Compute the ratio of the discharge of the solid phase to 
    the discharge of the liquid phase.

    qs/q = 4.2/(s-1) * i^1.6 * * (1 - theta_cr*(s-1)*Dm/h/i)

    Parameters
    ----------
    h : float
        The water depth
    i : float
        The slope of the riverbed
    s : float
        The ration of the solid grains density to water density
    theta_cr : float
        The critical shear stress
    Dm : float
        The mean grain diameter
    
    Returns
    -------
    float
        Thr ratio of solid discharge to liquid discharge
    """
    return 4.2/(s - 1) * i**1.6 * (1 - theta_cr*(s-1)*Dm/h/i)


DIR = Path(__file__).parent
shields = pd.read_csv(DIR / "shields-data.csv")
vanrijn = pd.read_csv(DIR / "shields-vanrijn.csv")


class ShieldsDiagram:
    """
    An object meant to facilitate the plotting of a Shields Diagram.

    >>> SD = ShieldsDiagram()
    >>> SD.plot(adim_shear, reynolds, adim_diam, *plot_args, **plot_kwargs)
    >>> fig, (axS, axVR) = SD.get_subplots()
    >>> plt.show()
    """
    def __init__(self,
                 figure=None,
                 axShields=None,
                 axVanRijn=None,
                 subplots_kw=None,
                 plot_kw=None,
                 *figure_args,
                 **figure_kwargs) -> None:
        """
        Object for plotting convenience. 
        Initialize it to show the frontiers and 
        the plot whatever is needed.

        Parameters
        ----------
        figure : matplotlib.pyplot.figure = None
        axShields : matplotlib.pyplot.Axes = None
            Axis to plot the Shields diagram on
        axVanRijn : matplotlib.pyplot.Axes = None
            Axis to plot the Shields (according ot Van Rijn) diagram on
        subplots_kw : dict
            Keyword arguments to pass to matplotlib.pyplot.subplots()
        plot_kw : dict
            Keyword arguments to pass to matplotlib.pyplot.plot
            for both Sheilds and VanRijn lines
        *figure_args
            Arguments to pass to matplotplib.pyplot.figure
        **figure_kwargs
            Keyword arguments to pass to matplotplib.pyplot.figure
        
        Attributes
        ----------
        figure : matplotlib.Figure
        axShields : matplotlib.axes.Axes
        axVanRijn : matplotlib.axes.Axes
        """
        subplots_kw = subplots_kw or dict()
        subplots_kw = dict(ncols=2, gridspec_kw=dict(wspace=0)) | subplots_kw
        if axShields is None and axVanRijn is None:
            figure = plt.figure(*figure_args, **figure_kwargs)
            axShields, axVanRijn = figure.subplots(**subplots_kw)

        if plot_kw is None:
            plot_kw = dict()
        axShields.loglog(shields.reynolds, shields.shear, **plot_kw)
        axVanRijn.loglog(vanrijn.diameter, vanrijn.shear, **plot_kw)

        axShields.set_title("Diagramme de Shields")
        axShields.set_xlabel(r"Reynolds $R=u_\ast d/\nu$")
        axShields.set_ylabel(r"$\Theta=\tau/(\rho g[s-1]d)$""\nCisaillement critique adimensionnel")

        axVanRijn.set_title("Selon Van Rijn")
        axVanRijn.yaxis.tick_right()
        axVanRijn.yaxis.set_label_position('right')
        axVanRijn.set_xlabel(r"Diamètre adimentionnel $d_\ast=d\cdot\sqrt[3]{(s-1)g/\nu^2}$")
        axVanRijn.set_ylabel("Cisaillement critique adimensionnel\n"r"$\Theta=\tau/(\rho g[s-1]d)$")

        self.figure = figure
        self.axShields = axShields
        self.axVanRijn = axVanRijn

    def plot(self,
             adim_shear: Iterable,
             reynolds: Iterable = None,
             adim_diam: Iterable = None,
             *plot_args,
             **plot_kwargs):
        """
        Plot on the dedicated axes in the Shields or Van Rijn diagrams.

        Parameters
        ----------
        adim_shear : Iterable
            The adimensional shear
        reynolds : Iterable
            The Reynolds numbers
        adim_diam : Iterable
            The adimensional grain diameters
        
        Returns
        -------
        List[matplotlib.lines.Line2D]
            The plotted lines
        """
        linesS = linesVR = []
        if reynolds is not None:
            linesS = self.axShields.plot(reynolds, adim_shear, *plot_args, **plot_kwargs)
        if adim_diam is not None:
            linesVR = self.axVanRijn.plot(adim_diam, adim_shear, *plot_args, **plot_kwargs)
        return linesS + linesVR

    def get_subplots(self):
        """
        Returns
        -------
        matplotlib.pyplot.figure
        matplotlib.pyplot.Axes
            Shields diagram
        matplotlib.pyplot.Axes
            Van Rijn diagram
        """
        return self.fig, (self.ax1, self.ax2)


def main():

    from profile import Profile

    df = pd.read_csv(DIR / 'profile.csv')
    section = Profile(
        df['Dist. cumulée [m]'],
        df['Altitude [m s.m.]'],
        33,
        12/100
    )

    grains = pd.read_csv("hydrogibs/fluvial/grains.csv")
    granulometry = interp1d(grains["Tamisats [%]"], grains["Diamètre des grains [cm]"])
    d16, d50, d90 = granulometry((16, 50, 90))
    sd = ShieldsDiagram()
    S, P = section.df.query("300 <= Q <= 1600")[["S", "P"]].to_numpy().T
    for d in (d16, d50, d90):
        Rh = S/P
        shear = rho*g*Rh*0.12/100
        diam = np.full_like(Rh, fill_value=d/100)
        r = reynolds(np.sqrt(shear/rho), diam)
        s = adimensional_shear(shear, diam, rho_s)
        d = adimensional_diameter(diam, rho_s)
        sd.plot(s, r, d)
    sd.figure.show()
    plt.show()


if __name__ == "__main__":
    main()
