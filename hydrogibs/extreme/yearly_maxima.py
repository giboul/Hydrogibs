"""
Definition of the 'YearlyMaxima' object, allowing to fit a poisson relation
(GEV or Generalized Extreme Value) between frequency and rainfall or discharge.

The GEV function is as follows:
P(x; loc, scale, shape) = exp[-(1 + shape*[x-loc]/scale)**(-1/shape)]
with x being the random variable
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from typing import List, Literal, Callable


def mse(a: np.ndarray, b: np.ndarray) -> float:  # mean squared error
    """
    Mean Squared Error: default error function used for the curve fitting

    Parameters
    ----------
    a : numpy array (1D)
    b : numpy array (1D)

    Returns
    -------
    float
        the mean squared error between 'a' and 'b'
    """
    return ((a-b)**2).sum()


def GEV(x: np.ndarray,
        loc: float,
        scale: float,
        shape: float) -> np.ndarray:
    """
    Definition of the GEV function

    Parameters
    ----------
    x : float | np.ndarray (1D)
        The quantile
    loc : float
        The first parameter of the GEV function
    scale : float
        The second parameter of the GEV function
    shape :
        The third parameter of the GEV function

    Returns
    -------
    float | np.ndarray
        The non-exceedance probability
    """
    if shape == 0:
        return np.exp(-np.exp(-(x-loc)/scale))
    else:
        return np.exp(-(1 + shape*(x-loc)/scale)**(-1/shape))


def GEV_inv(P: np.ndarray,
            loc: float,
            scale: float,
            shape: float) -> np.ndarray:
    """
    Definition of the inverted GEV function

    Parameters
    ----------
    P : float | np.ndarray (1D)
        The probability of non-exceedance array
    loc : float
        The first parameter of the GEV function
    scale : float
        The second parameter of the GEV function
    shape :
        The third parameter of the GEV function

    Returns
    -------
    np.ndarray
        The quantiles
    """
    if shape == 0:
        return loc - scale*np.log(-np.log(P))
    else:
        return loc - scale/shape * (1-(-np.log(P))**-shape)


transform_dict = {
    "probability": {
        "return period": lambda P: 1/(1-P),
        "gumbel": lambda P: -np.log(-np.log(P)),
        "probability": lambda P: P
    },
    "gumbel": {
        "return period": lambda u: 1/(1-np.exp(-np.exp(-u))),
        "probability": lambda u: np.exp(-np.exp(-u)),
        "gumbel": lambda u: u
    },
    "return period": {
        "probability": lambda T: 1-1/T,
        "gumbel": lambda T: -np.log(-np.log(1-1/T)),
        "return period": lambda T: T
    }
}


def fit_gumbel_params(x: np.ndarray):
    """
    Calculates the most fitting parameters according to the
    mean squared error when the parameter 'shape' is zero.
    The point of using this function is to avoid using
    an iterative method to find the parameters.

    Paramters
    ---------
    x: np.ndarray
        the quantiles to fit

    Returns
    -------
    Tuple[float, float]
        The first two GEV parameters (the third one is always zero)
    """
    scale = np.sqrt(6)*x.std()/np.pi
    loc = x.mean() - 0.577*scale

    return loc, scale


def fit_GEV_params(quantiles: np.ndarray,
                   probabilities: np.ndarray,
                   error_func: Callable = mse,
                   **optikwargs):
    """
    Through scipy's 'minimize' function, fnd the most fitting parameters
    of the function GEV: probabilities -> quantiles function.

    Parameters
    ----------
    quantiles : np.ndarray
        The quantiles to fit
    probabilities : np.ndarray
        The corresponding probabilities
    error_func : Callable
        The error function to use in the minimization
    """
    if 'x0' not in optikwargs:
        optikwargs['x0'] = *fit_gumbel_params(quantiles), 0
    if 'bounds' not in optikwargs:
        optikwargs['bounds'] = default_bounds(quantiles)

    def error(params):
        return error_func(quantiles, GEV_inv(probabilities, *params))

    return minimize(error, **optikwargs).x


def default_bounds(quantiles,
                   lower_xi=-float("inf"),
                   upper_xi=float("inf")) -> List[List[float]]:
    r"""
    The default bounds to use for the fitting of the GEV parameters

    Parameters
    ----------
    quantiles : np.ndarray
        The quantiles to fit
    lower_xi : float
        The lower bound for the third parameter (shape or $\xi$)
    upper_xi : float
        The upper bound for the third parameter (shape or $\xi$)

    Returns
    -------
    List[Tuple[float]]
        The bounds as [
            [lower_loc, upper_loc],
            [lower_scale, uppper_scale],
            [lower_xi, upper_xi]
        ]
    """
    max = quantiles.max()
    min = quantiles.min()
    return [[min, max],
            [0.0, 2*(max-min)],
            [lower_xi, upper_xi]]


class YearlyMaxima:

    kinds = ('frechet', 'gumbel', 'weibull')

    def __init__(self,
                 values: np.ndarray,
                 error_func: Callable = None) -> None:
        """
        Object for an easy computation of the most fitting parameters for each
        of the three laws (Fréchet, Gumbel & Weibull) and easy predictions.
        """
        df = pd.DataFrame(values, columns=["Q"]).sort_values("Q")
        n = df.Q.size
        df["rank"] = np.arange(1, n+1)
        df["prob"] = (df["rank"] - 0.28)/(n + 0.28)
        df["T"] = 1/(1 - df.prob)
        df["u"] = -np.log(-np.log(df.prob))
        self.dataframe = df

        if error_func is None:
            error_func = mse
            self.gumbel_params = *fit_gumbel_params(df.Q), 0

        bounds = default_bounds(df.Q, lower_xi=0)
        self.frechet_params = fit_GEV_params(
            df.Q,
            df.prob,
            x0=self.gumbel_params,
            bounds=bounds,
            error_func=error_func
        )
        bounds[2] = (-float("inf"), 0)
        self.weibull_params = fit_GEV_params(
            df.Q,
            df.prob,
            x0=self.gumbel_params,
            bounds=bounds,
            error_func=error_func
        )

        error_dict = {
            kind: error_func(getattr(self, kind)(df.Q), self.p)
            for kind in self.kinds
        }
        best_kind = min(error_dict, key=error_dict.get)
        self.best = getattr(self, best_kind)

    def frechet(self, probabilities):
        return GEV(probabilities, *self.frechet_params)

    def frechet_inv(self, quantiles):
        return GEV_inv(quantiles, *self.frechet_params)

    def gumbel(self, quantiles):
        return GEV(quantiles, *self.gumbel_params)

    def gumbel_inv(self, quantiles):
        return GEV_inv(quantiles, *self.gumbel_params)

    def weibull(self, quantiles):
        return GEV(quantiles, *self.weibull_params)

    def weibull_inv(self, probabilities):
        return GEV_inv(probabilities, *self.weibull_params)

    def predict(self,
                array,
                kind: Literal["frechet",
                              "gumbel",
                              "weibul",
                              "best"] = "best",
                input: Literal['probability',
                               'gumbel',
                               'return period'] = 'probability'):
        func = getattr(self, kind)
        probs = transform_dict[input]['probability'](array)
        return func(probs)

    def predict_inv(self,
                    quantiles,
                    kind: Literal[
                        "frechet",
                        "gumbel",
                        "weibul",
                        "best"] = "best",
                    output: Literal['probability',
                                    'gumbel',
                                    'return period'] = 'probability'):
        func = getattr(self, f"{kind}_inv")
        probs = func(quantiles)
        return transform_dict["probability"][output](probs)

    @property
    def u(self):
        return self.dataframe.u

    @property
    def p(self):
        return self.dataframe.prob

    @property
    def T(self):
        return self.dataframe["T"]

    @property
    def Q(self):
        return self.dataframe.Q

    def plot(
        self,
        kind: Literal["probability", "gumbel", "return period"] = "gumbel",
        fig=None, ax=None,
        show=False,
        style="ggplot"
    ):

        with plt.style.context(style):
            with plt.style.context({'font.family': 'monospace'}):

                if fig is None:
                    fig = plt.gcf()
                if ax is None:
                    ax = fig.subplots()

                x = transform_dict['probability'][kind](self.p)
                ax.scatter(x, self.Q, s=20, label="Empirique", zorder=4)

                _prob = np.linspace(self.p.min(), self.p.max(), num=1000)
                _x = transform_dict['probability'][kind](_prob)

                ax.plot(
                    _x,
                    self.frechet_inv(_prob),
                    label=rf"Fréchet $\xi={self.frechet_params[2]:.2f}$",
                    zorder=2
                )
                ax.plot(
                    _x,
                    self.weibull_inv(_prob),
                    label=rf"Weibull $\xi={self.weibull_params[2]:.2f}$",
                    zorder=1
                )
                ax.plot(
                    _x,
                    self.gumbel_inv(_prob),
                    label='Gumbel',
                    zorder=1
                )
                ax.legend()
                ax.set_xlabel(_xaxis_label[kind])
                ax.set_ylabel(
                    f"Quantiles des maxima\nannuels du débit (m$^3$/s)"
                )
                fig.tight_layout()
                if show:
                    plt.show()
        return fig, ax


_xaxis_label = {
    "gumbel": "Variable réduite de Gumbel "
              rf"$u=-\log\left(-\log\left(1-\frac{{1}}{{T}}\right)\right)$",
    "probability": "Probabilité de non-dépassement",
    "return period": "Période de retour (années)"
}


def main():
    df = pd.read_csv("hydrogibs/extreme/débits_mensuels_reyran.csv")
    df.t = pd.to_datetime(df.t, format="%Y-%m-%d %H:%M:%S")

    ym = YearlyMaxima(df.Q)
    ym.plot(kind='return period', show=True)


if __name__ == "__main__":
    main()
