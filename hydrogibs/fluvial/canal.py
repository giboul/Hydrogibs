from typing import Iterable, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl


if __name__ == "__main__":
    from hydrogibs.constants import g
else:
    from ..constants import g


def _new_df(sort_key='x', **kwargs):
    """
    Just for creating and sorting arrays faster and safer
    thx pandas
    """
    return pd.DataFrame(
        zip(*kwargs.values()), columns=[*kwargs.keys()]
    ).sort_values(sort_key)


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
    width = 0
    ###
    mask = (z_arr[1:] <= z) & (z_arr[:-1] <= z)
    dz = np.diff(z_arr)[mask]
    dx = np.diff(x_arr)[mask]
    zm = (z_arr[:-1] + z_arr[1:])[mask]
    length = np.sqrt(dx**2 + dz**2).sum()
    surface = ((z - zm/2) * dx).sum()
    # for x1, x2, z1, z2 in zip(x_arr[:-1], x_arr[1:], z_arr[:-1], z_arr[1:]):
    #     if z1 <= z and z2 <= z:
    #         length += np.sqrt((x2-x1)**2 + (z2-z1)**2)
    #         surface += (z - (z2+z1)/2) * (x2-x1)
    #         width += x2 - x1

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

        # 1. Store input data
        self.rawdata = _new_df(x=x, z=z)

        # 2. enhance coordinates
        _x, _z = twin_points(self.rawdata.x, self.rawdata.z)
        newdata = _new_df(x=_x, z=_z)
        self.data = _new_df(
            x=np.concatenate((self.rawdata.x, newdata.x)),
            z=np.concatenate((self.rawdata.z, newdata.z))
        )

        # 3. Reduce left and right boundaries
        _x, _z = strip_outside_world(self.data.x, self.data.z)
        self.data = _new_df(x=_x, z=_z)

        # Compute wet section's properties
        self.data["P"], self.data["S"], self.data["w"] = zip(*[
            polygon_properties(self.x, self.z, z)
            for z in self.z
        ])
        self.data["h"] = self.data.z - self.data.z.min()

    @property
    def x(self):
        return self.data.x

    @property
    def z(self):
        return self.data.z

    @property
    def h(self):
        return self.data.h

    @property
    def P(self):
        return self.data.P

    @property
    def S(self):
        return self.data.S

    @property
    def Rh(self):
        return self.data.Rh

    @property
    def Q(self):
        return self.data.Q

    def set_GMS_data(self, manning_strickler_coefficient: float, slope: float):
        """
        Set the Gauckler-Manning-Strickler discharges to the
        'data' DataFrame and return the entire DataFrame
        To get the discharge exclusively, get the 'Q' attribute

        Parameters
        ----------
        manning_strickler_coefficient : float
            As in v = K * Rh * i^0.5
        slope : float
            The slope of the energy line

        Returns
        -------
        pandas.DataFrame
            DataFrame containing all relevant data
        """
        self.K = manning_strickler_coefficient
        self.i = slope

        self.data["Q"] = self.data.S * GMS(
            self.K,
            self.data.S/self.data.P,
            self.i
        )

        return self

    def set_critical_data(self):

        x, z, h, S, P = self.data[
            ["x", "z", "h", "S", "P"]
        ][self.data.P > 0].to_numpy().T

        # critical values computing
        dS = S[2:] - S[:-2]
        dh = h[2:] - h[:-2]
        mask = ~ np.isclose(dS, 0, atol=1e-10, rtol=.0)
        dh_dS = np.full_like(x, None)
        dh_dS[1:-1][mask] = dh[mask]/dS[mask]

        # Q is computed from the derivative of S(h)
        # to avoid error accumulation with an integral
        Q = np.sqrt(g*S**3*dh_dS)

        self.critical_data = _new_df(x=x, z=z, h=h,
                                     S=S, P=P, Q=Q,
                                     sort_key='z')
        # remove invalid values
        self.critical_data = self.critical_data.loc[
            ~np.isnan(Q) | ~ (Q == 0)
        ]

        mask = np.isclose(dh_dS, 0)
        Fr = (Q[mask]**2/g/S[mask]**3/dh_dS[mask])
        if not np.isclose(Fr[~np.isnan(Fr)], 1, atol=10**-3).all():
            print("Critical water depths might not be representative")

        return self

    def set_speeds(self, speeds):

        x, z = self.rawdata.to_numpy().T

        w = np.diff(x)
        h = np.array([
            (h2+h1)/2 for h2, h1 in zip(z[1:], z[:-1])
        ])
        v = np.array([
            (v2+v1)/2
            for v2, v1 in zip(speeds[1:], speeds[:-1])
        ])
        q = w * h * v

        self.rawdata["qi"] = np.hstack(([0], q))
        self.rawdata["wi"] = np.hstack(([0], w))
        self.rawdata["hi"] = np.hstack(([0], h))
        self.rawdata["vi"] = np.hstack(([0], v))

        q = np.hstack(([0], q))

        return self

    def interp_P(self, h_array: np.ndarray) -> float:  # TODO !!!

        h_array = np.asarray(h_array)

        h, P = self.data[
            ["h", "P"]
        ].sort_values("h").drop_duplicates("h").to_numpy().T

        p = np.zeros_like(h_array)
        for i, h_interp in enumerate(h_array):
            # Checking if its within range
            mask = h >= h_interp
            if mask.all():
                p[i] = 0
                continue
            if not mask.any():
                p[i] = P[-1]

            # Find upper and lower bounds
            argsup = mask.argmax()
            arginf = argsup - 1
            # interpolate
            r = (h_interp - h[arginf]) / (h[argsup] - h[arginf])
            p[i] = P[arginf] + r * (P[argsup] - P[arginf])

        return p

    def interp_S(self, h_array: np.ndarray) -> float:  # TODO !!!

        h_array = np.asarray(h_array)

        h, w, S = self.data[
            ["h", "w", "S"]
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

    def interp_Q(self, h_array: float) -> float:
        h = np.asarray(h_array)
        S = self.interp_S(h)
        P = self.interp_P(h)
        Q = np.zeros_like(h)
        mask = ~np.isclose(P, 0)
        Q[mask] = S[mask] * GMS(self.K, S[mask]/P[mask], self.i)
        return Q

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
        if ax1 is None:
            ax1 = fig.add_subplot()
        if ax0 is None:
            ax0 = fig.add_subplot()
            ax0.patch.set_visible(False)

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
        if "Q" in self.data:
            df = self.data.sort_values('z')
            ax1.plot(df.Q, df.h, '--b',
                     label="$y_0$ (hauteur d'eau)")
        if hasattr(self, "critical_data"):
            ax1.plot(self.critical_data.Q, self.critical_data.h,
                     '-.', label='$y_{cr}$ (hauteur critique)')
        ax1.set_xlabel('Débit [m$^3$/s]')
        ax1.set_ylabel("Hauteur d'eau [m]")
        ax0.grid(False)

        # plotting 'RG' & 'RD'
        x01 = (1-0.05)*self.rawdata.x.min() + 0.05*self.rawdata.x.max()
        x09 = (1-0.95)*self.rawdata.x.min() + 0.95*self.rawdata.x.max()
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
        return fig, ax0, ax1

    def plot_terrain_data(self, ax=None, **subplotkwargs):

        ax = ax or plt.subplot(**subplotkwargs)

        position, profondeur, vitesse = self.rawdata[["x", "z", "vi"]].to_numpy().T

        xp = position[0]
        hp = profondeur[0]
        vp = vitesse[0]
        vmin = min(vitesse)
        vmax = max(vitesse)

        cmap = plt.get_cmap('Blues')
        for x, h, v in zip(position[1:], profondeur[1:], vitesse[1:]):
            c = cmap((v-vmin)/(vmax-vmin))
            ax.fill((xp, x, x, xp), (0, 0, h, hp),
                    color=c,
                    edgecolor=c)
            xp, hp, vp = x, h, v

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, aspect=7)
        # plt.plot(position, profondeur, '-o', c='k', mfc='w', mew=2, ms=5)
        ax.invert_yaxis()

        return ax, cb


DIR = Path(__file__).parent


def test_Section():

    df = pd.read_csv(DIR / 'profile.csv')
    section = Section(
        df['Dist. cumulée [m]'],
        df['Altitude [m s.m.]'],
    ).set_critical_data().set_GMS_data(33, 0.12/100)
    with plt.style.context('ggplot'):
        section.plot(show=True)
    
    h = np.linspace(section.h.min(), section.h.max(), 1000)
    df = pd.DataFrame(
        zip(h, section.interp_S(h), section.interp_P(h), section.interp_Q(h)),
        columns=('h', 'S', 'P', 'Q')
    )
    df.to_csv(DIR / 'hydraulic_data.csv', index=False)


def test_measures():

    with plt.style.context("ggplot"):
        DIRo = Path("/home/axel/Documents/EPFL/Eco Morhpologie/")
        for type in ("naturel", "artificiel"):
            for i in range(1, 5):
                sheet = f"{type[:3]}{i}"
                name = f"{type} {i}"
                fig = plt.figure(figsize=(6, 2.5))
                df = pd.read_excel(DIR / 'Mesures.xlsx', sheet_name=sheet).fillna(0)
                section = Section(x=df.position-df.position.min(), z=df.profondeur)
                ax, cb = section.set_speeds(df.vitesse).plot_terrain_data()
                ax.set_facecolor('none')
                cb.set_label("Vitesse [m/s]")
                ax.set_xlabel("Position [m]")
                ax.set_ylabel("Profondeur [cm]")
                hi, vi = section.rawdata[["hi", "vi"]].to_numpy().T
                hvar = np.std(hi)/hi.mean()
                vvar = np.std(vi)/vi.mean()
                ax.set_title(# f"Tronçon {name}\n"
                             # f"$\sigma_h={hsig:.0f}$ cm"
                             # "   "
                             # f"$\sigma_v={vsig:.0f}$ cm/s\n"
                             f" $CV_{{v}}={vvar:.2f}$"
                             "   "
                             f" $CV_{{h}}={hvar:.2f}$")
                plt.tight_layout()
                plt.savefig(DIRo / f"section_mesure_{sheet}.pdf", bbox_inches="tight")
                fig.show()
        plt.show()


if __name__ == "__main__":
    # test_measures()
    test_Section()
