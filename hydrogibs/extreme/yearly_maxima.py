import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from typing import Iterable, List, Tuple, Literal


def mse(a, b):  # mean squared error
    return ((a-b)**2).sum()


def poisson(P: np.ndarray, loc: float, scale: float, shape: float):
    if shape == 0:
        return loc - scale*np.log(-np.log(P))
    else:
        return loc - scale/shape * (1-(-np.log(P))**-shape)


def fit_gumbel(values: np.ndarray):

    scale = np.sqrt(6)*values.std()/np.pi
    loc = values.mean() - 0.577*scale

    return loc, scale


def fit_poisson(quantiles, probabilities, x0=None, bounds=None):

    if x0 is None:
        x0 = *fit_gumbel(quantiles), 0
    if bounds is None:
        bounds = default_bounds(quantiles)

    solution = minimize(
        lambda params: mse(poisson(probabilities, *params), quantiles),
        x0=x0,
        bounds=bounds
    )

    return solution.x


def default_bounds(quantiles,
                   lower_xi=-float("inf"),
                   upper_xi=float("inf")) -> List[Tuple[float, float]]:
    max = quantiles.max()
    min = quantiles.min()
    return [(min, max),
            (0.0, 2*(max-min)),
            (lower_xi, upper_xi)]


class YearlyMaxima:

    kinds = ('frechet', 'gumbel', 'weibull')

    def __init__(self, values) -> None:

        df = pd.DataFrame(values, columns=["Q"]).sort_values("Q")
        n = df.Q.size
        df["rank"] = np.arange(1, n+1)
        df["prob"] = (df["rank"] - 0.28)/(n + 0.28)
        df["T"] = 1/(1 - df.prob)
        df["u"] = -np.log(-np.log(df.prob))
        self.df = df

        self.gumbel_params = *fit_gumbel(df.Q), 0
        bounds = default_bounds(df.Q, lower_xi=0)
        self.frechet_params = fit_poisson(df.Q, df.prob, x0=self.gumbel_params, bounds=bounds)
        bounds[2] = (-float("inf"), 0)
        self.weibull_params = fit_poisson(df.Q, df.prob, x0=self.gumbel_params, bounds=bounds)

        # [setattr(self, k, lambda p: poisson(p, *getattr(self, f"{k}_params"))) for k in self.kinds]

        error_dict = {kind: mse(getattr(self, kind)(self.df.prob), df.Q) for kind in self.kinds}
        best_kind = min(error_dict, key=error_dict.get)
        self.best = getattr(self, best_kind)

    def frechet(self, probabilities):
        return poisson(probabilities, *self.frechet_params)

    def gumbel(self, probabilities):
        return poisson(probabilities, *self.gumbel_params)

    def weibull(self, probabilities):
        return poisson(probabilities, *self.weibull_params)

    def predict(self, probabailities, kind: Literal["frechet", "gumbel", "weibul", "best"] = "best"):
        return getattr(self, kind)(probabailities)

    @property
    def u(self):
        return self.df.u

    @property
    def p(self):
        return self.df.prob

    @property
    def T(self):
        return self.df["T"]

    @property
    def Q(self):
        return self.df.Q

    def plot(self, fig=None, ax=None, show=False, style="ggplot"):

        with plt.style.context(style):
            with plt.style.context({'font.family': 'monospace'}):

                if fig is None:
                    fig = plt.gcf()
                if ax is None:
                    ax = fig.subplots()

                ax.scatter(self.u, self.Q, s=20, label="avant 2010", zorder=2)
                _prob = np.linspace(self.p.min(), self.p.max(), num=1000)
                _u = -np.log(-np.log(_prob))
                ax.plot(_u, self.frechet(_prob), label=rf"Fréchet $\xi={self.frechet_params[2]:.2f}$", zorder=1)
                ax.plot(_u, self.weibull(_prob), label=rf"Weibull $\xi={self.weibull_params[2]:.2f}$", zorder=0)
                ax.plot(_u, self.gumbel(_prob), label='Gumbel', zorder=0)
                ax.legend()
                ax.set_xlabel(rf"Variable réduite de Gumbel $u=-\log\left(-\log\left(1-\frac{{1}}{{T}}\right)\right)$")
                ax.set_ylabel(f"Quantiles des maxima\nannuels du débit (m$^3$/s)")
                fig.tight_layout()
                if show:
                    plt.show()
        return fig, ax


def main():
    df = pd.read_csv("hydrogibs/extreme/dfy.csv")
    df.t = pd.to_datetime(df.t, format="%Y-%m-%d %H:%M:%S")

    ym = YearlyMaxima(df.Q)
    fig, ax = ym.plot()

    f2010 = df.sort_values("Q").t.dt.year > 2010
    ax.scatter(ym.u[f2010], ym.Q[f2010], s=20, zorder=3)
    plt.show()


if __name__ == "__main__":
    main()
