from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    from hydrogibs.constants import g
else:
    from ..constants import g


def find_roots(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-16
) -> np.ndarray[float]:
    """
    Function for quickly finding the roots from an array with
    an interpolation (to avoid non-horizontal water tables)
    """
    y[y == 0] = -eps
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)


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

    for i, z in enumerate(z_arr):
        x_intersection = find_roots(x_arr, z_arr - z)
        if x_intersection.size:
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

    # find the index opposite to zmax
    if z[right_max] < z[left_max]:
        # find first value below zmax starting from the bottom (argmin)
        left_max = argmin - (left & (z <= z[right_max]))[argmin::-1].argmin()+1
    else:
        # find first value below zmax starting from the bottom again
        right_max = argmin + (right & (z <= z[left_max]))[argmin:].argmin()-1
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
      _   ___         _
    _/ \_/...\       / \
           ↑  \_____/   \
        This surface should not contribute

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

    def __init__(
        self,
        x: Iterable,  # position array from left to right river bank
        z: Iterable,  # altitude array from left to right river bank
        K: float,  # Manning-Strickler coefficient
        i: float  # River bed's slope
    ) -> None:

        def new_df(x: np.ndarray, z: np.ndarray):
            return pd.DataFrame(
                zip(x, z), columns=["x", "z"]
            ).sort_values('x')

        # 1. Store input data
        self.rawdata = new_df(x, z)

        # 2. enhace coordinates
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

        self.K = K
        self.i = i

    @property
    def x(self):
        return self.data.x

    @property
    def z(self):
        return self.data.z

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
        elevation, pyplot axis
        discharge - water depth, pyplot axis
        """
        if fig is None:
            fig = plt.figure()
        if ax0 is None:
            ax0 = fig.add_subplot()
        if ax1 is None:
            ax1 = fig.add_subplot()
            ax1.patch.set_visible(False)

        # plotting input bed coordinates
        lxz, = ax0.plot(self.rawdata.x, self.rawdata.z, '-ok',
                        label='Profil en travers')
        # potting framed coordinates (the ones used for computations)
        ax0.plot(self.x, self.z, '-o', mfc='w',
                 zorder=lxz.get_zorder(),
                 label='Profil en travers')

        # bonus wet section example
        if h is not None:
            poly_data = self.data[self.data.z <= h + self.data.z.min()]
            polygon, = ax0.fill(
                poly_data.x, poly_data.z,
                alpha=0.6,
                label='Section mouillée'
            )
        ax0.set_xlabel('Distance profil [m]')
        ax0.set_ylabel('Altitude [m.s.m.]')

        # positionning axis labels on right and top
        ax0.xaxis.tick_top()
        ax0.xaxis.set_label_position('top')
        ax0.yaxis.tick_right()
        ax0.yaxis.set_label_position('right')

        # normal water depths according to GMS
        data = self.data.sort_values("z")
        z, S, P = data[["z", "S", "P"]][data.P > 0].to_numpy().T

        h = z - z.min()
        Q = S * GMS(self.K, S/P, self.i)

        # critical values computing
        dh_dS = (h[2:] - h[:-2])/(S[2:] - S[:-2])
        Q_cr = np.sqrt(g*S[1:-1]**3*dh_dS)
        mask = Q_cr <= Q.max()
        Q_cr = Q_cr[mask]
        h_cr = h[1:-1][mask]
        S_cr = S[1:-1][mask]
        P_cr = P[1:-1][mask]

        Fr_cr = Q_cr**2/g/S_cr**3/dh_dS[mask]
        if not np.isclose(Fr_cr, 1, atol=10**-3).all():
            print("Critical water depths might not be representative")

        # plotting water depths
        ax1.plot(Q, h, label="$y_0$ (hauteur d'eau)")
        ax1.plot(Q_cr, h_cr, label='$y_{cr}$ (hauteur critique)')
        ax1.set_xlabel('Débit [m$^3$/s]')
        ax1.set_ylabel('Profondeur [m]')
        ax1.grid(False)

        # plotting 'RG' & 'RD'
        x01 = (1-0.05)*self.rawdata.x.min() + 0.05*self.rawdata.x.max()
        x09 = Q.max()
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
