import numpy as np
from matplotlib import pyplot as plt
from ..constants import g
from .misc import water_depth_solutions, conjugate


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
