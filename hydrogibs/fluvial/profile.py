"""
Script for estimating the h-Q relationship 
from a given profile (according to GMS). 

The object 'Profile' stores the hydraulic data as 
a pandas.DataFrame and creates a complete diagram 
with the .plot() method.

The following friction laws are supported here :
    - Gauckler-Manning-Strickler
    - Darcy
    - Chézy

Run script along with the following files to test:
    - profile.csv
    - closedProfile.csv
    - minimalProfile.csv
It will plot three diagrams with :
    - Limits enclosing the problem
    - The water_depth-discharge relation
    - The water_depth-critical_discharge relation
"""
from typing import Iterable, Tuple
from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import click


g = 9.81


def GMS(K: float, Rh: float, i: float) -> float:
    """
    The Manning-Strickler equation

    Q = K * S * Rh^(2/3) * sqrt(i)

    Parameters
    ----------
    K : float
        The Manning-Strickler coefficient
    Rh : float
        The hydraulic radius, area/perimeter or width
    Js : float
        The slope of the riverbed

    Return
    ------
    float
        The discharge according to Gauckler-Manning-Strickler
    """
    return K * Rh**(2/3) * i**0.5


def equivalent_laws(Rh: float,
                    K: float = None,
                    C: float = None,
                    f: float = None) -> Tuple[float]:

    Rh = np.array(Rh)
    Rh[np.isclose(Rh, 0)] = None

    if sum(x is not None for x in (K, C, f)) != 1:
        raise ValueError("Specify exactly one of (K, C, f)")

    if K is not None:
        C = K * Rh**(1/6)
        f = 8*g / (K**2 * Rh**(1/3))
    if C is not None:
        K = C / Rh**(1/6)
        f = 8 * g / C**2
    if f is not None:
        K = (8*g/f)**0.5 / Rh**(1/6)
        C = (8*g/f)**0.5

    def array(a):
        if isinstance(a, (float, int)):
            a = np.full_like(Rh, a)
        a[np.isnan(a)] = 0
        return a

    K, C, f = map(array, (K, C, f))

    return K, C, f


def twin_points(x_arr: Iterable, z_arr: Iterable) -> Tuple[np.ndarray]:
    """
    Duplicate an elevation to every crossing of its level and the (x, z) curve.
    This will make for straight water tables when filtering like this :
    >>> z_masked = z[z <= z[ix]]  # array with z[ix] at its borders
    Thus, making the cross-section properties (S, P, B) easily computable.
    _                          ___
    /|     _____              ////
    /|    //////\            /////
    /+~~~+-------o~~~~~~~~~~+/////
    /|__//////////\        ///////
    ///////////////\______////////
    //////////////////////////////
    Legend:
         _
        //\ : ground
        ~ : water table
        o : a certain point given by some pair of (x, z)
        + : the new points created by this function

    Parameters
    ----------
    x : Iterable
        the horizontal coordinates array
    y : Iterable
        the vertical coordinates array

    Return
    ------
    np.ndarray
        the enhanced x-array
    np.ndarray
        the enhanced y-array
    """
    x_arr = np.array(x_arr, dtype=np.float32)
    z_arr = np.array(z_arr, dtype=np.float32)
    points = np.vstack((x_arr, z_arr)).T

    # to avoid looping over a dynamic array
    new_x = np.array([], dtype=np.float32)
    new_z = np.array([], dtype=np.float32)
    new_i = np.array([], dtype=np.int32)

    for i, line in enumerate(zip(points[:-1], points[1:]), start=1):

        (x1, z1), (x2, z2) = line

        if abs(z1-z2) < 1e-10:
            continue

        add_z = np.sort(z_arr[(min(z1, z2) < z_arr) & (z_arr < max(z1, z2))])
        if z2 < z1:  # if descending,
            add_z = add_z[::-1]  # reverse order
        add_x = x1 + (x2 - x1) * (add_z - z1)/(z2 - z1)
        add_i = np.full_like(add_z, i, dtype=np.int32)

        new_x = np.hstack((new_x, add_x))
        new_z = np.hstack((new_z, add_z))
        new_i = np.hstack((new_i, add_i))

    x = np.insert(x_arr, new_i, new_x)
    z = np.insert(z_arr, new_i, new_z)

    return x, z


def strip_outside_world(x: Iterable, z: Iterable) -> Tuple[np.ndarray]:
    """
    Return the same arrays without the excess borders
    (where the flow section width is unknown).

    If this is not done, the flow section could extend
    to the sides and mess up the polygon.

    This fuction assumes that twin_points has just been applied.

    Example of undefined profile:

             _
            //\~~~~~~~~~~~~~~~~~~  <- Who knows where this water table ends ?
           ////\          _
    ______//////\        //\_____
    /////////////\______/////////
    /////////////////////////////
    Legend:
         _
        //\ : ground
        ~ : water table

    Parameters
    ----------
    x : Iterable
        Position array from left to right
    z : Iterable
        Elevation array

    Return
    ------
    np.ndarray (1D)
        the stripped x
    np.ndarray(1D)
        the stripped y
    """
    x = np.array(x, dtype=np.float32)  # so that indexing works properly
    z = np.array(z, dtype=np.float32)
    ix = np.arange(x.size)  # indexes array
    argmin = z.argmin()  # index for the minimum elevation
    left = ix <= argmin  # boolean array inidcatinf left of the bottom
    right = argmin <= ix  # boolean array indicating right

    # Highest framed elevation (avoiding profiles with undefined borders)
    zmax = min(z[left].max(), z[right].max())
    right_max_arg = argmin + (z[right] == zmax).argmax()
    left_max_arg = argmin - (z[left] == zmax)[::-1].argmax()
    right[right_max_arg+1:] = False
    left[:left_max_arg] = False

    return x[left | right], z[left | right]


def polygon_properties(
    x_arr: Iterable,
    z_arr: Iterable,
    z: float
) -> Tuple[float]:
    """
    Return the polygon perimeter and area of the formed polygons.

    Parameters
    ----------
    x : Iterable
        x-coordinates
    y : Iterable
        y-coordinates
    z : float
        The z threshold (water table elevation)

    Return
    ------
    float
        Permimeter of the polygon
    float
        Surface area of the polygon
    float
        Length of the water table
    """
    x_arr = np.array(x_arr, dtype=np.float32)
    z_arr = np.array(z_arr, dtype=np.float32)

    mask = (z_arr[1:] <= z) & (z_arr[:-1] <= z)
    zm = (z_arr[:-1] + z_arr[1:])[mask]/2
    dz = np.diff(z_arr)[mask]
    dx = np.diff(x_arr)[mask]

    length = np.sqrt(dx**2 + dz**2).sum()
    surface = np.abs(((z - zm) * dx).sum())
    width = np.abs(dx.sum())

    return length, surface, width


def hydraulic_data(x: Iterable, z: Iterable) -> pd.DataFrame:
    """
    Derive relation between water depth and discharge (Manning-Strickler)

    Parameters
    ----------
    x : Iterable
        x (transversal) coordinates of the profile. 
        These values will be sorted.
    z : Iterable
        z (elevation) coordinates of the profile. 
        It will be sorted according to x.

    Return
    ------
    pandas.DataFrame
        x : x-coordinates
        z : z-coordinates
        P : wet perimeter
        S : wet surface
        B : dry perimeter
        h : water depth
        Qcr : critical discharge
        Q : discharge (if GMS computed)
    """
    # Compute wet section's properties
    P, S, B = np.transpose([polygon_properties(x, z, zi) for zi in z])
    h = z - z.min()

    Rh = np.full_like(P, None)
    Rh[np.isclose(S, 0)] = 0
    mask = ~ np.isclose(P, 0)
    Rh[mask] = S[mask] / P[mask]

    # Compute h_cr-Qcr
    Qcr = np.full_like(B, None)
    mask = ~ np.isclose(B, 0)
    Qcr[mask] = np.sqrt(g*S[mask]**3/B[mask])

    return pd.DataFrame.from_dict(dict(
        h=h, P=P, S=S, Rh=Rh, B=B, Qcr=Qcr
    ))


def profile_diagram(
    x: Iterable,
    z: Iterable,
    h: Iterable,
    Q: Iterable,
    Qcr: Iterable,
    fig=None,
    axes=None
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot riverbed cross section and Q(h) in a sigle figure

    Parameters
    ----------
    h : float
        Water depth of stream cross section to fill
    show : bool
        wether to show figure or not
    fig, (ax0, ax1)
        figure and axes on which to draw (ax0: riverberd, ax1: Q(h))

    Returns
    -------
    pyplot figure
        figure containing plots
    pyplot axis
        profile coordinates transversal position vs. elevation
    pyplot axis
        discharge vs. water depth
    """
    if fig is None:
        fig = plt.figure()
    if axes is None:
        ax1 = fig.add_subplot()
        ax0 = fig.add_subplot()
        ax0.patch.set_visible(False)

    x = np.array(x, dtype=np.float32)
    z = np.array(z, dtype=np.float32)
    h = np.array(h, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    Qcr = np.array(Qcr, dtype=np.float32)

    l1, = ax0.plot(x, z, '-ok',
                   mfc='w', lw=3, ms=5, mew=1,
                   label='Profil en travers utile')

    ax0.set_xlabel('Distance profil [m]')
    ax0.set_ylabel('Altitude [m.s.m.]')

    # positionning axis labels on right and top
    ax0.xaxis.tick_top()
    ax0.xaxis.set_label_position('top')
    ax0.yaxis.tick_right()
    ax0.yaxis.set_label_position('right')

    # plotting water depths
    ix = h.argsort()  # simply a sorting index
    l2, = ax1.plot(Q[ix], h[ix], '--b', label="$y_0$ (hauteur d'eau)")
    l3, = ax1.plot(Qcr[ix], h[ix], '-.', label='$y_{cr}$ (hauteur critique)')
    ax1.set_xlabel('Débit [m$^3$/s]')
    ax1.set_ylabel("Hauteur d'eau [m]")
    ax0.grid(False)

    # plotting 'RG' & 'RD'
    ztxt = z.mean()
    ax0.text(x.min(), ztxt, 'RG')
    ax0.text(x.max(), ztxt, 'RD', ha='right')

    # match height and altitude ylims
    ax1.set_ylim(ax0.get_ylim() - z.min())

    # common legend for both axes
    lines = (l1, l2, l3)
    labels = [line.get_label() for line in lines]
    ax0.legend(lines, labels)

    ax1.dataLim.x0 = 0
    ax1.autoscale_view()

    return fig, (ax0, ax1)


class Profile(pd.DataFrame):
    """
    An :func:`~pandas.DataFrame` object.

    Attributes
    ----------
    x : pd.Series
        x-coordinates 
        (horizontal distance from origin)
    z : pd.Series
        z-coordinates (altitudes)
    h : pd.Series
        Water depths
    P : pd.Series
        Wtted perimeter
    S : pd.Series
        Wetted area
    Rh : pd.Series
        Hydraulic radius
    Q : pd.Series
        Discharge (GMS)
    Q : pd.Series
        Critical discharge
    K : float
        Manning-Strickler coefficient
    Js : float
        bed's slope

    Methods
    -------
    plot(h: float = None)
        Plots a matplotlib diagram with the profile,
        the Q-h & Q-h_critical curves and a bonus surface from h
    interp_Q(h: Iterable)
        Returns an quadratic interpolation of the discharge (GMS)
    """

    def __init__(
        self,
        x: Iterable,  # position array from left to right river bank
        z: Iterable,  # altitude array from left to right river bank
        **fric_kwargs
    ) -> None:
        """
        Initialize :func:`~hydraulic_data(x, z, K, Js)` and set the friction law Js

        Parameters
        ----------
        x: Iterable
            position array from left to right river bank
        z: Iterable
            altitude array from left to right river bank
        K: float = None
            The manning-strickler coefficient
        C: float = None
            The Chézy coefficient
        f: float = None
            The Darcy-Weisbach coefficient
        Js: float = None
            The riverbed's slope
        """

        x, z = twin_points(x, z)
        x, z = strip_outside_world(x, z)
        df = pd.DataFrame.from_dict(dict(x=x, z=z))
        hd = hydraulic_data(x, z)
        df = pd.concat((df, hd), axis="columns")

        super().__init__(df)

        if fric_kwargs:
            Js = fric_kwargs.pop("Js")
            if isinstance(Js, float):
                Js = np.full_like(self.x, Js)
            K, C, f = equivalent_laws(self.Rh, **fric_kwargs)
            self["v"] = GMS(K, self.Rh, Js)
            self["Q"] = self.S * self.v

            self["K"] = K
            self["Js"] = Js

    def interp_K(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.K)(h_array)

    def interp_Js(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.Js)(h_array)

    def interp_B(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.B)(h_array)

    def interp_P(self, h_array: Iterable) -> np.ndarray:
        return interp1d(self.h, self.P)(h_array)

    def interp_S(self, h_array: Iterable) -> np.ndarray:
        """
        Quadratic interpolation of the surface. 
        dS = dh*dB/2 where B is the surface width

        Parameters
        ----------
        h_array : Iterable
            Array of water depths

        Returns
        -------
        np.ndarray
            The corresponding surface area
        """

        h, B, S = self[
            ["h", "B", "S"]
        ].sort_values("h").drop_duplicates("h").to_numpy().T

        s = np.zeros_like(h_array)
        for i, h_interp in enumerate(h_array):
            # Checking if h_interp is within range
            mask = h >= h_interp
            if mask.all():
                s[i] = 0
                continue
            if not mask.any():
                s[i] = float("nan")
                continue

            # Find lower and upper bounds
            argsup = mask.argmax()
            arginf = argsup - 1
            # interpolate
            r = (h_interp - h[arginf]) / (h[argsup] - h[arginf])
            Bi = r * (B[argsup] - B[arginf]) + B[arginf]
            ds = (h_interp - h[arginf]) * (Bi + B[arginf])/2
            s[i] = S[arginf] + ds

        return s

    def interp_Q(self, h_array: Iterable) -> np.ndarray:
        """
        Interpolate discharge from water depth with
        the quadratic interpolation of S.

        Parameters
        ----------
        h_array : Iterable
            The water depths array.

        Return
        ------
        np.ndarray
            The corresponding discharges
        """
        h = np.array(h_array, dtype=np.float32)
        S = self.interp_S(h)
        P = self.interp_P(h)
        Q = np.zeros_like(h)
        mask = ~np.isclose(P, 0)
        Q[mask] = S[mask] * GMS(
            self.interp_K(h)[mask],
            S[mask]/P[mask],
            self.interp_Js(h)[mask]
        )
        return Q

    def interp_Qcr(self, h_array: Iterable) -> np.ndarray:
        """
        Interpolate critical discharge from water depth.

        Parameters
        ----------
        h_array : Iterable
            The water depths array.

        Return
        ------
        np.ndarray
            The corresponding critical discharge
        """
        Qcr = np.full_like(h_array, None)
        B = self.interp_B(h_array)
        S = self.interp_S(h_array)
        mask = ~ np.isclose(B, 0)
        Qcr[mask] = np.sqrt(g*S[mask]**3/B[mask])
        return Qcr

    def plot(self, interp_num=1000, *args, **kwargs) -> Tuple[Figure, Tuple[plt.Axes]]:
        """Call :func:`~profile_diagram(self.x, self.z,  self.h, self.Q, self.Qcr)` 
        and update the lines with the interpolated data."""
        fig, (ax1, ax2) = profile_diagram(
            self.x, self.z, self.h, self.Q, self.Qcr,
            *args, **kwargs
        )

        l1, l2 = ax2.get_lines()
        h = np.linspace(self.h.min(), self.h.max(), interp_num)
        l1.set_data(self.interp_Q(h), h)
        l2.set_data(self.interp_Qcr(h), h)

        return fig, (ax1, ax2)


DIR = Path(__file__).parent


def test_Section(reverse=False):

    df = pd.read_csv(DIR / 'profile.csv')
    if reverse:
        df['Altitude [m s.m.]'] = np.array(df['Altitude [m s.m.]'])[::-1]
    profile = Profile(
        df['Dist. cumulée [m]'],
        df['Altitude [m s.m.]'],
        f=1,
        Js=0.12/100
    )

    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = profile.plot()
        ax1.plot(df['Dist. cumulée [m]'],
                 df['Altitude [m s.m.]'],
                 '-o', ms=8, c='gray', zorder=0,
                 lw=3, label="Profil complet")
        ax2.dataLim.x1 = profile.Q.max()
        ax2.autoscale_view()
        ax2.set_ylim(ax1.get_ylim()-profile.z.min())
        fig.show()


def test_ClosedSection():

    df = pd.read_csv(DIR / 'closedProfile.csv')
    r = 10
    K = 33
    Js = 0.12/100
    profile = Profile(
        (df.x+1)*r, (df.z+1)*r,
        K=K, Js=Js
    )

    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = profile.plot()
        ax2.dataLim.x1 = profile.Q.max()
        ax2.autoscale_view()

        # Analytical solution
        theta = np.linspace(1e-10, np.pi)
        S = theta*r**2 - r**2*np.cos(theta)*np.sin(theta)
        P = 2*theta*r
        Q = K*(S/P)**(2/3)*S*Js**0.5
        h = r * (1-np.cos(theta))
        ax2.plot(Q, h, alpha=0.5, label="$y_0$ (analytique)")

        ax1.legend(loc="upper left").remove()
        ax2.legend(loc=(0.2, 0.6)).get_frame().set_alpha(1)
        ax2.set_ylim(ax1.get_ylim()-profile.z.min())
        fig.show()


def test_minimal():
    df = pd.read_csv(DIR / "minimalProfile.csv")
    with plt.style.context("ggplot"):
        prof = Profile(df.x, df.z, K=33, Js=0.12/100)
        fig, (ax1, ax2) = prof.plot()
        ax1.plot(df.x, df.z, '-o', ms=8, lw=3, c='gray', zorder=0)
        ax2.dataLim.x1 = prof.Q.max()
        ax2.set_ylim(ax1.get_ylim()-prof.z.min())
        ax2.autoscale_view()
        fig.show()


def csv_to_csv(input_file: str,
               output_file: str,
               plot: bool) -> None:

    if output_file is not None:
        output_file = f"{Path(input_file).stem}-processed-hydrogibs.csv"

    df = pd.read_csv(input_file)
    K = 33
    i = 0.12/100
    profile = Profile(df.x, df.z, K, i)
    profile.to_csv(output_file, index=False)

    if plot is not None:
        with plt.style.context("ggplot"):
            fig, (ax1, ax2) = profile.plot()
            ax1.plot(df.x, df.z, '-o', ms=8, lw=3, c='gray', zorder=0)
            ax2.dataLim.x1 = profile.Q.max()
            ax2.autoscale_view()
            plt.show()


@click.command()
@click.option('-i', '--input-file')
@click.option('-o', '--output-file')
@click.option('-p', '--plot', default=False)
@click.option('-t', '--test', default=False, is_flag=True)
def main(input_file: str,
         output_file: str,
         plot: bool,
         test: bool) -> None:

    if input_file is not None:
        csv_to_csv(input_file, output_file, plot)
        exit()

    if test or (__name__ == "__main__" and not plot and not test and not input_file):
        test_minimal()
        test_Section()
        test_Section(reverse=True)
        test_ClosedSection()
        plt.show()


if __name__ == "__main__":
    main()

