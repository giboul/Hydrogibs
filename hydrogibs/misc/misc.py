import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult, minimize, fsolve
from warnings import warn
from ..constants import g


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
        warn(f"\ncrupedix: Catchment area is not within recommended range:\n\t"
             f"{S:.3e} not in [1,4 * 10^3 km^2 - 52 * 10^3 km^2]")
    return R * S**0.8 * (Pj10/80)**2


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


def charge_hydraulique(h, v, z=.0, g=g):
    return h + z + v**2/(2*g)


def critical_depth(Q, Sfunc, eps=0.1, h0=1, g=g):

    def deriv(h):
        return (Sfunc(h+eps) - Sfunc(h-eps)) / (2*eps)

    return float(fsolve(
        lambda h: Q**2/(g*Sfunc(h)**3) * deriv(h) - 1,
        x0=h0
    ))


def water_depth_solutions(H, Q, Sfunc, z=0, g=g,
                          eps=10**-3, num=100, **optkwargs):

    hcr = critical_depth(Q, Sfunc)

    def head_diff(h):
        return np.abs(charge_hydraulique(h, Q/Sfunc(h), z) - H)

    xsub = float(
        minimize(
            head_diff,
            x0=hcr*0.5,
            bounds=((10**-2, hcr-eps),),
            **optkwargs
        ).x
    )
    xsup = float(
        minimize(
            head_diff,
            x0=hcr*1.5,
            bounds=((hcr+eps, None),),
            **optkwargs
        ).x
    )

    return (xsub, xsup)


def besse(x, y, slope, hn, hc):
    ydot = slope * (1 - (hn / y) ** 3) / (1 - (hc / y) ** 3)
    return ydot


def besseEuler(x, y0, slope, hn, hc, stop=False):
    y = np.full_like(x, float('nan'))
    y[0] = y0
    for n in range(0, len(x)-1):
        v = y[n] + besse(x[n], y[n], slope, hn, hc)*(x[n+1] - x[n])
        if v >= hc and stop:
            break
        y[n+1] = v
    return y


def conjugate(q, h, g=g):
    return 0.5 * h * (np.sqrt(
        8*(q/h**1.5/np.sqrt(g))**2 + 1
    ) - 1)


class Ressaut:

    def __init__(self,
                 q, i1, i2, h0, x0, xt,
                 p=0., ms_K=None, chezy_C=None, g=g,
                 dx=None, num=None
                 ) -> None:

        self.q = q
        self.i1, self.i2 = i1, i2
        self.h0 = h0
        self.p = p
        self.x0, self.xt = x0, xt
        self.lawname, self.fk, self.flaw = self._set_flaw(chezy_C, ms_K)
        self.g = g
        self.dx, self.num = dx, num

        self.x, self.h, self.position = self.calculate()

    def calculate(self):

        hc = (self.q**2/self.g)**(1/3)
        hn1 = self.flaw(self.q, self.i1, self.fk)
        hn2 = self.flaw(self.q, self.i2, self.fk)

        x1, x2, dx = self._xarrays(
            self.x0,
            self.xt,
            self.num,
            self.dx
        )
        h1 = besseEuler(x1, self.h0, self.i1, hn1, hc)
        h2 = besseEuler(x2, h1[-1], self.i2, hn2, hc, stop=True)

        slice = ~np.isnan(h2)
        h2 = h2[slice]
        x2 = x2[slice]

        x3 = np.arange(self.xt, 1000 + x2.max() + dx, step=dx)

        Hcr = 3/2 * hc + self.p
        _, hb = water_depth_solutions(
            H=Hcr,
            Q=self.q,
            Sfunc=lambda h: h,
            tol=10**-3
        )

        h3 = besseEuler(x3[::-1], hb, self.i2, hn2, hc)[::-1]
        position = x2[np.argmin(np.abs(conjugate(self.q, h2) - h3[:len(h2)]))]

        self.result = (
            np.concatenate((x1, x2, x3)),
            np.concatenate((h1, h2, h3)),
            position
        )

        slice2 = x2 <= position
        slice3 = x3 >= position
        self.results_verbose = (
            (x1, x2[slice2], x3[slice3]),
            (h1, h2[slice2], h3[slice3]),
            (x1, x2[~slice2], x3[~slice3]),
            (h1, h2[~slice2], h3[~slice3])
        )

        return self.result

    def diagram(self, style='ggplot', show=False, **subplotkwargs):

        (
            (x1, x2, x3),
            (h1, h2, h3),
            (x1x, x2x, x3x),
            (h1x, h2x, h3x)
        ) = self.results_verbose

        xmax = 2 * x2x.max() if x2x.size else 4 * x2[-1]

        x = np.concatenate((x1, x2, x3))
        bed = -self.i1 * x
        bed[x >= self.xt] = (
            - self.xt * self.i1
            - self.i2 * (x[x >= self.xt] - self.xt)
        )
        bed -= bed[-1]
        bed1 = bed[:x1.size]
        bed2 = bed[x1.size:x1.size+x2.size]
        bed3 = bed[x1.size+x2.size:]
        bed3 = bed3[x3 <= xmax]
        h3 = h3[x3 <= xmax]
        x3 = x3[x3 <= xmax]

        bed = bed[x <= xmax]
        x = x[x <= xmax]

        with plt.style.context(style):
            fig, ax = plt.subplots(**subplotkwargs)

            ax.plot(
                (x2.max(), x3.min()),
                (bed2[-1] + h2.max(), bed2[-1] + h3.min()),
                'k'
            )
            ax.plot(x1, bed1 + h1, label='coursier')
            ax.plot(x2, bed2 + h2, label='supercritique')
            ax.plot(x3, bed3 + h3, label='subcritique')
            ax.plot(x, bed + (self.q**2/self.g)**(1/3),
                    ':', lw=1, label="h$_{cr}$")
            bedx = bed[x1.size+x2.size:x1.size+x2.size+x2x.size]
            ax.plot(x2x, bedx + h2x, '-.k', alpha=0.4)
            ax.plot(x2x, bedx + conjugate(self.q, h2x), '--k', alpha=0.4)
            plt.fill_between(x, bed, bed.min()*0.9,
                             color='k', edgecolor='none',
                             alpha=0.8, lw=2, label='lit')
            plt.fill_between(
                x, bed, bed + np.concatenate((h1, h2, h3)),
                color='b', edgecolor='none', alpha=0.2, zorder=1
            )
            ax.set_xlabel("x (m)")
            ax.set_ylabel("h (m.s.m.)")

            eps, k = 0.1, 4
            # ax.set_xlim(self.x0, k*self.position)
            # ax.set_ylim(
            #     (1-eps)*h1.min(),
            #     (1+eps) * h3[x3 < k*self.position].max()
            # )
            ax.legend(loc="center right")

        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def _set_flaw(self, chezy_C, ms_K):
        if (chezy_C and ms_K) or (chezy_C or ms_K) is None:
            raise ValueError("Give only one of 'chezy_C' and 'ms_K'")
        elif chezy_C is not None:
            return 'Chézy', chezy_C, lambda q, i, C: (q/C/np.sqrt(i))**(2/3)
        elif ms_K is not None:
            return 'Manning', ms_K, lambda q, i, K: (q/K/np.sqrt(i))**(3/5)

    def _xarrays(self, x0, xt, num, dx):
        xf = x0 + 100 * (xt-x0)
        if dx is None and num is None:
            dx = 0.25
        if dx:
            x1 = np.arange(x0, xt+dx, step=dx)
            x2 = np.arange(xt, xf+dx, step=dx)
        elif num:
            x1 = np.linspace(x0, xt, num=num, endpoint=True)
            x2 = np.linspace(xt, xf, num=num, endpoint=True)
            dx = x1[1] - x1[0]
        return x1, x2, dx


if __name__ == "__main__":
    r = Ressaut(
        q=10, i1=0.05, i2=0.002, p=0.5,
        h0=2, ms_K=30, x0=0, xt=10, dx=0.25
    )
    r.diagram(show=True, figsize=(10, 5))
