from typing import Iterable, Tuple
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib import pyplot as plt


g = 9.81


def _df(**kwargs):
    """Just to shorten the code"""
    return pd.DataFrame.from_dict(dict(kwargs))


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
    i : float
        The slope of the riverbed
    
    Return
    ------
    float
        The discharge according to Gauckler-Manning-Strickler
    """
    return K * Rh**(2/3) * i**0.5


def twin_points(x_arr: Iterable, z_arr: Iterable) -> Tuple:
    """
    Duplicate an elevation to every crossing of its level and the (x, z) curve.
    This will make for straight water tables when filtering like this :
    >>> z_masked = z[z <= z[some_index]]  # array with z[some index] at its borders
    Thus, making the section properties (S, P, B) easily computable.

    _                          ___
    /|     _____              ////
    /|    //////\            /////
    /o~~~o~~~~~~~o~~~~~~~~~~o/////
    /|__//////////\        ///////
    ///////////////\______////////
    //////////////////////////////

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
    x_arr = np.asarray(x_arr)
    z_arr = np.asarray(z_arr)
    points = np.vstack((x_arr, z_arr)).T

    # to avoid looping over a dynamic array
    new_x = np.array([])
    new_z = np.array([])
    new_i = np.array([], dtype=np.int32)

    for i, ((x1, z1), (x2, z2)) in enumerate(zip(points[:-1], points[1:]), start=1):

        add_z = np.sort(z_arr[(min(z1, z2) < z_arr) & (z_arr < max(z1, z2))])
        if z2 < z1:
            add_z = add_z[::-1]  # if descending, reverse order
        add_i = np.full_like(add_z, i, dtype=np.int32)
        add_x = x1 + (x2 - x1) * (add_z - z1)/(z2 - z1)  # interpolation

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

    Example of undefined section:

             _
            //\~~~~~~~~~~~~~~~~~~  <- Who knows where this water table ends ?
           ////\          _
    ______//////\        //\_____
    /////////////\______/////////
    /////////////////////////////

    Parameters
    ----------
    x : np.ndarray (1D)
        Position array from left to right
    z : np.ndarray (1D)
        Elevation array

    Return
    ------
    np.ndarray (1D)
        the stripped x
    np.ndarray(1D)
        the stripped y
    """
    x = np.asarray(x)  # so that indexing works properly
    z = np.asarray(z)
    ix = np.arange(x.size)  # indexes array
    argmin = z.argmin()  # index for the minimum elevation
    left = ix <= argmin  # boolean array inidcatinf left of the bottom
    right = argmin <= ix  # boolean array indicating right

    # Highest framed elevation (avoiding sections with undefined borders)
    left_max = z[left].argmax()
    right_max = z[right].argmax() + argmin

    # strip left to the highest framed elevation
    candidates = (left & (z <= z[right_max]))[argmin::-1]
    if not candidates.all():
        left_max = argmin - candidates.argmin()+1

    # strip right to the highest framed elevation
    candidates = (right & (z <= z[left_max]))[argmin:]
    if not candidates.all():
        right_max = argmin + candidates.argmin()-1

    left[:left_max] = False
    right[right_max+1:] = False

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
    x_arr = np.asarray(x_arr)
    z_arr = np.asarray(z_arr)

    mask = (z_arr[1:] <= z) & (z_arr[:-1] <= z)
    zm = (z_arr[:-1] + z_arr[1:])[mask]/2
    dz = np.diff(z_arr)[mask]
    dx = np.diff(x_arr)[mask]

    length = np.sqrt(dx**2 + dz**2).sum()
    surface = np.abs(((z - zm) * dx).sum())
    width = np.abs(dx.sum())

    return length, surface, width


class Section:
    """
    An object storing and plotting hydraulic data about the given cross-section

    Attributes
    ----------
    rawdata : pd.DataFrame
        DataFrame containing given x and z coordinates
    newdata : pd.DataFrame
        DataFrame with more points
    df : pd.DataFrame
        concatenation of rawdata & newdata
    K : float
        Manning-Strickler coefficient
    i : float
        bed's slope

    Properties
    ----------
    x : pd.Series
        Shortcut for the enhanced coordinates
        self.data.x
    z : pd.Series
        Shortcut for the enhanced altitudes
        self.data.z
    h : pd.Series
        Shortcut for the enhanced water depths 
        self.data.z
    P : pd.Series
        Shortcut for the wet perimeter
    S : pd.Series
        Shortcut for the wet area
    Rh : pd.Series
        Shortcut for the hydraulic radius
    Q : pd.Series
        Shortcut for the dicharge (GMS)

    Methods
    -------
    plot(h: float = None)
        Plots a matplotlib diagram with the profile,
        the Q-h & Q-h_critical curves and a bonus surface from h
    Q(h: float)
        Returns an interpolated value of the discharge (GMS)
    h(Q: float)
        Returns an interpolated value of the water depth
    """
    def __init__(
        self,
        x: Iterable,  # position array from left to right river bank
        z: Iterable,  # altitude array from left to right river bank
    ) -> None:
        """
        This object is meant to derive water depth to discharge relations
        and plot them along with the profile in a single diagram.

        Parameters
        ----------
        x : Iterable
            x (transversal) coordinates of the profile. 
            These values will be sorted.
        z : Iterable
            z (elevation) coordinates of the profile. 
            It will be sorted according to x.
        
        Attributes
        ----------
        rawdata : pandas.DataFrame
            The input x and z values.
        df : pandas.DataFrame (after Section.preprocess())
            The enahnced and stripped data with 
            wet perimeter, wet surface and surface width 
            (also GMS after Section.compute_GMS_data() 
            and critical discharge after Section.compute_critical_data()).
        x : x-coordinates (after Section.preprocess())
        z : z-coordinates (after Section.preprocess())
        P : wet perimeter
        S : wet surface
        B : dry perimeter
        h : water depth
        """

        # 1. Store input data
        self.rawdata = _df(x=x, z=z)
        # 2. enhance and strip coordinates
        self = self.preprocess()
        # 3. Compute wet section's properties
        self = self.compute_geometry()

    def preprocess(self):
        """
        Ehance the data by duplicating every altitude as many times as
        it has antecedents in the z(x) interpolation. Then strip the data
        to points with a defined wet section only.

        Set attribute
        -------------
        df : pandas.DataFrame
            enhanced data from section.rawdata
        x : x-coorinates
        z : elevation
        """

        x, z = strip_outside_world(self.rawdata.x, self.rawdata.z)
        x, z = twin_points(x, z)
        self.df = _df(x=x, z=z)

        return self
    
    def compute_geometry(self):
        """
        Compute the wet section's perimeter, area and width (and height).

        Set attribute
        -------------
        P : wet perimeter
        S : wet surface
        B : dry perimeter
        h : water depth
        """
        self.df["P"], self.df["S"], self.df["B"] = zip(*[
            polygon_properties(self.x, self.z, z)
            for z in self.z
        ])
        self.df["h"] = self.df.z - self.df.z.min()

        return self

    def compute_GMS_data(self, manning_strickler_coefficient: float, slope: float):
        """
        Set the Gauckler-Manning-Strickler discharges to the
        'df' DataFrame and return the entire DataFrame
        To get the discharge exclusively, get the 'Q' attribute

        Parameters
        ----------
        manning_strickler_coefficient : float
            As in v = K * Rh * i^0.5
        slope : float
            The slope of the energy line

        Set attribute
        -------------
        Section
            Object containing all relevant data in the
            "df" (pandas.DataFrame) attribute
        """
        self.K = manning_strickler_coefficient
        self.i = slope
    
        self.df["v"] = GMS(self.K, self.S/self.P, self.i)
        self.df["Q"] = self.S * self.v

        return self

    def compute_critical_data(self):
        """
        Compute the critical discharges for every possible water depth.
        
        Q = sqrt(g*S^3*dh/dS)

        Set attribute
        -------------
        Qcr : The critical discharges
        """
        # dS / dh = B
        self.df["Qcr"] = np.sqrt(g*self.S**3/self.B)

        return self

    @property
    def x(self):
        return self.df.x

    @property
    def z(self):
        return self.df.z

    @property
    def h(self):
        return self.df.h

    @property
    def P(self):
        return self.df.P

    @property
    def S(self):
        return self.df.S

    @property
    def B(self):
        return self.df.B

    @property
    def Rh(self):
        return self.df.Rh

    @property
    def v(self):
        return self.df.v

    @property
    def Q(self):
        return self.df.Q

    @property
    def Qcr(self):
        return self.df.Qcr

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
        h_array = np.asarray(h_array)

        h, w, S = self.df[
            ["h", "B", "S"]
        ].sort_values("h").drop_duplicates("h").to_numpy().T

        s = np.zeros_like(h_array)
        for i, h_interp in enumerate(h_array):
            # Checking if its within range
            mask = h >= h_interp
            if mask.all():
                s[i] = 0
                continue
            if not mask.any():
                s[i] = S[-1]

            # Find lower and upper bounds
            argsup = mask.argmax()
            arginf = argsup - 1
            # interpolate
            r = (h_interp - h[arginf]) / (h[argsup] - h[arginf])
            wi = r * (w[argsup] - w[arginf]) + w[arginf]
            ds = (h_interp - h[arginf]) * (wi + w[arginf])/2
            s[i] = S[arginf] + ds

        return s
    
    def interp_Qcr(self, h_array: Iterable) -> np.ndarray:
        return np.sqrt(g*self.interp_S(h_array)**3/self.interp_B(h_array))

    def interp_Q(self, h_array: Iterable) -> np.ndarray:
        """
        Interpolate discharge from water depth with
        the quadratic interpolation of S.

        Parameters
        ----------
        h_array : Iterable
            The water depths array.
        
        Returns
        -------
        np.ndarray
            The corresponding discharges
        """
        h = np.asarray(h_array)
        S = self.interp_S(h)
        P = self.interp_P(h)
        Q = np.zeros_like(h)
        mask = ~np.isclose(P, 0)
        Q[mask] = S[mask] * GMS(self.K, S[mask]/P[mask], self.i)
        return Q

    def interp_h(self, Q_array: Iterable) -> np.ndarray:
        return NotImplementedError(
            "Yeah I'm not sure how to do this (surjective function)"
        )

    def plot(self, h: float = None,
             fig=None, ax0=None, ax1=None, show=False):
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
        if ax1 is None:
            ax1 = fig.add_subplot()
        if ax0 is None:
            ax0 = fig.add_subplot()
            ax0.patch.set_visible(False)

        # plotting input bed coordinates
        lxz, = ax0.plot(self.rawdata.x, self.rawdata.z, '-o',
                        color='gray', lw=3, ms=8, mew=1,
                        label='Profil en travers complet')
        # potting framed coordinates (the ones used for computations)
        ax0.plot(self.x, self.z, '-ok',
                 mfc='w', lw=3, ms=5, mew=1,
                 zorder=lxz.get_zorder(),
                 label='Profil en travers utile')

        # bonus wet section example
        if h is not None:
            poly_data = self.df[self.df.z <= h + self.df.z.min()]
            polygon, = ax0.fill(
                poly_data.x, poly_data.z,
                linewidth=0,
                alpha=0.3, color='b',
                label='Section mouillée',
                zorder=0
            )
        ax0.set_xlabel('Distance profil [m]')
        ax0.set_ylabel('Altitude [m.s.m.]')

        # positionning axis labels on right and top
        ax0.xaxis.tick_top()
        ax0.xaxis.set_label_position('top')
        ax0.yaxis.tick_right()
        ax0.yaxis.set_label_position('right')

        # plotting water depths
        df = self.df.dropna().sort_values('Qcr')
        if "Qcr" in df:
            ax1.plot(df.Qcr, df.h, '-.', label='$y_{cr}$ (hauteur critique)')
        df = self.df.sort_values('z')
        if "Q" in df:
            ax1.plot(df.Q, df.h, '--b', label="$y_0$ (hauteur d'eau)")
        ax1.set_xlabel('Débit [m$^3$/s]')
        ax1.set_ylabel("Hauteur d'eau [m]")
        ax0.grid(False)

        # plotting 'RG' & 'RD'
        x01 = (1-0.05)*self.rawdata.x.min() + 0.05*self.rawdata.x.max()
        x09 = (1-0.9)*self.rawdata.x.min() + 0.9*self.rawdata.x.max()
        ztxt = self.rawdata.z.mean()
        ax0.text(x01, ztxt, 'RG')
        ax0.text(x09, ztxt, 'RD')

        # match height and altitude ylims
        ax1.set_ylim(ax0.get_ylim() - self.z.min())

        # common legend
        lines = (*ax0.get_lines(), *ax1.get_lines())
        labels = [line.get_label() for line in lines]
        ax0.legend(lines, labels)

        # showing
        # fig.tight_layout()
        if show:
            plt.show()
        return fig, (ax0, ax1)


DIR = Path(__file__).parent


def test_Section():

    df = pd.read_csv(DIR / 'profile.csv')
    section = Section(
        df['Dist. cumulée [m]'],
        df['Altitude [m s.m.]'],
    ).compute_GMS_data(33, 0.12/100).compute_critical_data()
    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = section.plot()
        # ax2.dataLim.x1 = section.Q.max()
        # ax2.autoscale_view()
        # # Quadratic interpolation
        # h = np.linspace(section.h.min(), section.h.max(), 1000)
        # ax2.plot(section.interp_Qcr(h), h)
        # ax2.plot(section.interp_Q(h), h)
        # ax2.plot(df.Q, df.h)
        fig.show()


def test_ClosedSection():

    df = pd.read_csv(DIR / 'closedProfile.csv')
    r = 10
    K = 33
    i = 0.12/100
    section = Section(
        (df.x+1)*r, (df.z+1)*r,
    ).compute_GMS_data(K, i).compute_critical_data()

    with plt.style.context('ggplot'):
        fig, (ax1, ax2) = section.plot()
        ax2.dataLim.x1 = section.Q.max()
        ax2.autoscale_view()

        theta = np.linspace(1e-10, np.pi)
        S = theta*r**2 - r**2*np.cos(theta)*np.sin(theta)
        P = 2*theta*r
        Q = K*(S/P)**(2/3)*S*(i)**0.5
        h = r * (1-np.cos(theta))
        ax2.plot(Q, h, alpha=0.5, label="$y_0$ (analytique)")
        ax1.legend(loc="upper left").remove()
        ax2.legend(loc=(0.2, 0.6)).get_frame().set_alpha(1)
        # # Quadratic interpolation
        # h = np.linspace(section.h.min(), section.h.max(), 1000)
        # ax2.plot(section.interp_Qcr(h), h)
        # ax2.plot(section.interp_Q(h), h)
        fig.show()


if __name__ == "__main__":
    # test_measures()
    test_Section()
    test_ClosedSection()
    plt.show()
