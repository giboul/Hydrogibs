from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


if __name__ == "__main__":
    from hydrogibs.constants import g
else:
    from ..constants import g


def find_roots(
    x: Iterable,
    y: Iterable,
    eps: float = 1e-5
) -> np.ndarray[float]:
    """
    Function for quickly finding the roots from an array with
    an interpolation (to avoid non-horizontal water tables)

    Parameters
    ----------
    x : Iterable
    y : Iterable

    Returns
    -------
    x : np.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_zeros = x[y == 0]
    y[np.isclose(y, 0, atol=eps, rtol=0)] = -eps
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    x_roots = x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

    return np.concatenate((x_zeros, x_roots))


def GMS(K: float, Rh: float, i: float) -> float:
    """
    The Manning-Strickler equation

    Parameters
    ----------
    K : float
        The Manning-Strickler coefficient
    Rh : float
        The hydraulic radius, area/perimeter or width
    i : float
        The slope of the riverbed
    """
    return K * Rh**(2/3) * i**0.5


def twin_points(x_arr: np.ndarray, z_arr: np.ndarray) -> Tuple[np.ndarray]:
    """
    Duplicates a point to every crossing of its level and the (x, z) curve

    Parameters
    ----------
    x : np.ndarray
        the horizontal coordinates array
    y : np.ndarray
        the vertical coordinates array

    Returns
    -------
    np.ndarray
        the enhanced x-array
    np.ndarray
        the enhanced y-array
    """
    x_arr = np.asarray(x_arr)  # so that indexing works properly
    z_arr = np.asarray(z_arr)
    argmin = z_arr.argmin()
    x_mid = x_arr[argmin]
    new_x = np.array([])  # to avoid looping over a dynamic array
    new_z = np.array([])

    for i, (x, z) in enumerate(zip(x_arr, z_arr)):
        x_intersection = find_roots(x_arr, z_arr - z)
        if x_intersection.size:
            # remove trivial answer to avoid duplicate
            x_intersection = x_intersection[x_intersection != x]
            new_x = np.concatenate((new_x, x_intersection))
            new_z = np.concatenate((new_z, np.full_like(x_intersection, z)))

    return new_x, new_z


def strip_outside_world(x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray]:
    """
    Returns the same arrays without the excess borders
    (where the flow section width is unknown).

    If this is not done, the flow section could extend
    to the sides and mess up the polygon.

    \b
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

    Returns
    -------
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
    zmax = min(z[left_max], z[right_max])

    # strip left to the highest framed elevation
    candidates = (left & (z <= z[right_max]))[argmin::-1]
    if not candidates.all():
        left_max = argmin - candidates.argmin()+1

    # strip righ to the highest framed elevation
    candidates = (right & (z <= z[left_max]))[argmin:]
    if not candidates.all():
        right_max = argmin + candidates.argmin()-1

    left[:left_max] = False
    right[right_max+1:] = False

    return x[left | right], z[left | right]


def polygon_properties(
    x_arr: np.ndarray,
    z_arr: np.ndarray,
    z: float
) -> Tuple[float]:
    """
    Returns the polygon perimeter and area of the formed polygon.
    Particular attention is needed for the perimeter's calculation:

    \b
       _     ___           _
      //\~~~/~~~\~~~~~~~~~//\
    _////\_// ↑ /\       ////\
    ///////// ↑ //\_____//////\
    ///////// ↑ ///////////////\
              ↑
    This lenght should not contribute to the perimeter

    Parameters
    ----------
    x : np.ndarray
        x-coordinates
    y : np.ndarray
        y-coordinates
    z : float
        The z threshold (water table elevation)

    Returns
    -------
    float
        Permimeter of the polygon
    float
        Surface area of the polygon
    """
    x_arr = np.asarray(x_arr)
    z_arr = np.asarray(z_arr)

    mask = z_arr <= z
    length = 0
    surface = 0
    for x1, x2, z1, z2 in zip(x_arr[:-1], x_arr[1:], z_arr[:-1], z_arr[1:]):
        if z1 <= z and z2 <= z:
            length += np.sqrt((x2-x1)**2 + (z2-z1)**2)
            surface += (z - (z2+z1)/2) * (x2-x1)

    return length, surface


class Section:
    """
    An object storing and plotting hydraulic data about the given cross-section

    Attributes
    ----------
    rawdata : pd.DataFrame
        DataFrame containing given x and z coordinates
    newdata : pd.DataFrame
        DataFrame with more points
    data : pd.DataFrame
        concatenation of rawdata & newdata
    K : float
        Manning-Strickler coefficient
    i : float
        bed's slope

    Properties
    ----------
    x : pd.Series
        shortcut for self.data.x
    z : pd.Series
        shortcut for self.data.z

    Methods
    -------
    plot(h: float = None)
        Plots a matplotlib diagram with the profile,
        the Q-h & Q-h_critical curves and a bonus surface from h
    Q(h: float)
        Returns an interpolated value of the discharge
    h(Q: float)
        Returns an interpolated value of the water depth
    """
    def __init__(
        self,
        x: Iterable,  # position array from left to right river bank
        z: Iterable,  # altitude array from left to right river bank
        K: float,  # Manning-Strickler coefficient
        i: float  # River bed's slope
    ) -> None:
        """
        This object is meant to derive water depth to discharge relations
        and plot them along with the profile in a single diagram

        Parameters
        ----------
        x : Iterable
            x (transversal) coordinates of the profile
        z : Iterable
            z (elevation) coordinates of the profile
        K : float
            Manning-Strickler coefficient (might add more laws later)
        i : float
            slope of the riverbed
        """

        def new_df(x: np.ndarray, z: np.ndarray, sort_key='x', **kwargs):
            """
            Just for creating and sorting arrays faster and safer
            thx pandas
            """
            return pd.DataFrame(
                zip(x, z, *kwargs.values()), columns=["x", "z", *kwargs.keys()]
            ).sort_values(sort_key)

        self.K = K
        self.i = i

        # 1. Store input data
        self.rawdata = new_df(x, z)

        # 2. enhance coordinates
        self.newdata = new_df(*twin_points(self.rawdata.x, self.rawdata.z))
        self.data = new_df(
            np.concatenate((self.rawdata.x, self.newdata.x)),
            np.concatenate((self.rawdata.z, self.newdata.z))
        )

        # 3. Reduce left and right boundaries
        self.data = new_df(*strip_outside_world(self.data.x, self.data.z))

        self.data["P"], self.data["S"] = zip(*[
            polygon_properties(self.x, self.z, z)
            for z in self.z
        ])
        self.data["Q"] = self.data.S * GMS(
            self.K,
            self.data.S/self.data.P,
            self.i
        )
        self.data["h"] = self.data.z - self.data.z.min()

        zsorteddata = self.data.sort_values("z")
        self.interpolate_h_from_Q = interp1d(self.data.Q, self.data.h)
        self.interpolate_Q_from_h = interp1d(self.data.h, self.data.Q)

        x, z, h, Q, S, P = zsorteddata[
            ["x", "z", "h", "Q", "S", "P"]
        ][zsorteddata.P > 0].to_numpy().T
        self.zsorteddata = zsorteddata

        # critical values computing
        dS = S[2:] - S[:-2]
        dh = h[2:] - h[:-2]
        mask = ~np.isclose(dS, 0, atol=1e-10, rtol=.0)
        dh_dS = np.full_like(x, None)
        dh_dS[1:-1][mask] = dh[mask]/dS[mask]

        # Q is computed from the derivative of S(h)
        # to avoid error accumulation with an integral
        Q = np.sqrt(g*S**3*dh_dS)

        self.critical_data = new_df(x, z, h=h, S=S, P=P, Q=Q, sort_key='h')

        mask = np.isclose(dh_dS, 0)
        Fr = (Q[mask]**2/g/S[mask]**3/dh_dS[mask])
        if not np.isclose(Fr[~np.isnan(Fr)], 1, atol=10**-3).all():
            print("Critical water depths might not be representative")

    @property
    def x(self):
        return self.data.x

    @property
    def z(self):
        return self.data.z

    def Q(self, h: float):
        return self.interpolate_Q_from_h(h)

    def h(self, Q: float):
        return self.interpolate_h_from_Q(Q)

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
        fig, ax0, ax1
            figure and axes on which to draw (ax0: riverberd, ax1: Q(h))

        Returns
        -------
        pyplot figure
        pyplot axis
            profile coordinates transversal position vs. elevation
        pyplot axis
            discharge vs. water depth
        """
        if fig is None:
            fig = plt.figure()
        if ax0 is None:
            ax0 = fig.add_subplot()
        if ax1 is None:
            ax1 = fig.add_subplot()
            ax1.patch.set_visible(False)

        # plotting input bed coordinates
        lxz, = ax0.plot(self.rawdata.x, self.rawdata.z,
                        '-o', color='gray', lw=3,
                        label='Profil en travers complet')
        # potting framed coordinates (the ones used for computations)
        ax0.plot(self.x, self.z,
                 '-ok', mfc='w', lw=3,
                 zorder=lxz.get_zorder(),
                 label='Profil en travers utile')

        # bonus wet section example
        if h is not None:
            poly_data = self.data[self.data.z <= h + self.data.z.min()]
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
        ax1.plot(self.zsorteddata.Q, self.zsorteddata.h, '--',
                 label="$y_0$ (hauteur d'eau)")
        ax1.plot(self.critical_data.Q, self.critical_data.h,
                 '-.', label='$y_{cr}$ (hauteur critique)')
        ax1.set_xlabel('Débit [m$^3$/s]')
        ax1.set_ylabel('Profondeur [m]')
        ax1.grid(False)

        # plotting 'RG' & 'RD'
        x01 = (1-0.05)*self.rawdata.x.min() + 0.05*self.rawdata.x.max()
        x09 = (1-0.95)*self.rawdata.x.min() + 0.95*self.rawdata.x.max()
        ztxt = 1.2*self.rawdata.z.mean()
        ax0.text(x01, ztxt, 'RG')
        ax0.text(x09, ztxt, 'RD')

        # match height and altitude ylims
        ax1.set_ylim(ax0.get_ylim() - self.z.min())

        # common legend
        lines = (*ax0.get_lines(), *ax1.get_lines())
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels)

        # showing
        # fig.tight_layout()
        if show:
            return plt.show()
        return fig, ax0, ax1


if __name__ == "__main__":

    df = pd.read_csv('hydrogibs/test/fluvial/profile.csv')
    section = Section(
        df['Dist. cumulée [m]'],
        df['Altitude [m s.m.]'][::-1],
        K=33,
        i=0.12/100
    )
    with plt.style.context('ggplot'):
        # Diagramme conventionnel
        section.plot(show=True)
