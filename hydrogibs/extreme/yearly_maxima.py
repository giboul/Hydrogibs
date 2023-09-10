import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, Literal


def quantiles_and_probabilities(values, threshold=.0) -> Tuple[np.ndarray]:

    values = np.asarray(values)

    quantiles = np.sort(values[values >= threshold])
    ranks = np.arange(1, quantiles.size+1)
    probs = (ranks - 0.28)/(quantiles.size + 0.28)

    return quantiles, probs


def poisson(P: np.ndarray, loc: float, scale: float, shape: float):
    if shape == 0:
        return loc - scale*np.log(-np.log(P))
    else:
        return loc - scale/shape * (1-(-np.log(P))**-shape)


def default_xi_bounds(quantiles,
                      lower_xi=-float("inf"),
                      upper_xi=float("inf")):
    max = quantiles.max()
    min = quantiles.min()
    return ((min, max),
            (0.0, 2*(max-min)),
            (lower_xi, upper_xi))


def fit_annual_maxima(
    values: np.ndarray,
    threshold: float = 0,
    kind: Literal["gumbel", "weibull", "frechet"] = None,
    **optkwargs
):

    if (
        kind == "gumbel" or
        "bounds" in optkwargs and tuple(optkwargs["bounds"]) == (.0, .0)
    ):
        return *fit_gumbel(values, threshold), 0
    if "bounds" not in optkwargs:
        optkwargs["bounds"] = default_xi_bounds(
            values,
            lower_xi=0 if kind == "frechet" else -float("inf"),
            upper_xi=-0 if kind == "weibull" else float("inf")
        )
    if kind == "frechet":
        assert tuple(optkwargs["bounds"][2]) >= (0, 0)
    elif kind == "weibull":
        assert tuple(optkwargs["bounds"][2]) <= (0, 0)
    if "x0" not in optkwargs:
        optkwargs["x0"] = *fit_gumbel(values, threshold), 0

    return fit_poisson(values, threshold, **optkwargs)


def fit_poisson(values, threshold, **optkwargs):

    C, P = quantiles_and_probabilities(values, threshold)

    solution = minimize(
        lambda params: ((poisson(P, *params) - C)**2).sum(),
        **optkwargs
    )

    return solution.x


def fit_frechet(values, threshold=0,
                upper_xi=float("inf"),
                lower_xi=0, **optkwargs):

    if "bounds" not in optkwargs:
        optkwargs["bounds"] = default_xi_bounds(
            values,
            lower_xi=lower_xi,
            upper_xi=upper_xi
        )
    assert lower_xi >= 0 and (upper_xi is None or upper_xi > 0)
    if "x0" not in optkwargs:
        optkwargs["x0"] = *fit_gumbel(values, threshold), 0

    return fit_poisson(values, threshold, **optkwargs)


def fit_gumbel(values: np.ndarray, threshold: float = 0):

    values = values[values >= threshold]
    scale = np.sqrt(6)*values.std()/np.pi
    loc = values.mean() - 0.577*scale

    return loc, scale


def fit_weibull(values, threshold=0,
                lower_xi=-float("inf"),
                upper_xi=-0, **optkwargs):

    if "bounds" not in optkwargs:
        optkwargs["bounds"] = default_xi_bounds(
            values,
            lower_xi=lower_xi,
            upper_xi=upper_xi
        )
    if "x0" not in optkwargs:
        optkwargs["x0"] = *fit_gumbel(values, threshold), 0
    assert upper_xi <= 0 and (lower_xi is None or lower_xi < 0)

    return fit_poisson(values, threshold, **optkwargs)


_xaxis_transformation = {
    "gumbel": lambda p: -np.log(-np.log(p)),
    "probability": lambda p: p,
    "return period": lambda p: 1/(1-p)
}

_xaxis_label = {
    "gumbel": "Variable réduite de Gumbel "
              rf"$u=-\log\left(-\log\left(1-\frac{{1}}{{T}}\right)\right)$",
    "probability": "Probabilité de non-dépassement",
    "return period": "Période de retour (années)"
}


def analyse(annual_maxima, threshold=0,
            xaxis: Literal[
                "probability",
                "return period",
                "gumbel"
            ] = "gumbel",
            show=True, tight_layout=True,
            style="ggplot", font="monospace",
            _base_functions=False, **figkwargs):

    with plt.style.context(style):
        with plt.style.context({'font.family': font}):
            fig, ax = plt.subplots(**figkwargs)

            C, P = quantiles_and_probabilities(annual_maxima, threshold)
            if _base_functions:
                lg, sg, xg = *fit_gumbel(annual_maxima, threshold), 0.
                lf, sf, xf = fit_frechet(annual_maxima, threshold)
                lw, sw, xw = fit_weibull(annual_maxima, threshold)
            else:
                lg, sg, xg = fit_annual_maxima(annual_maxima, threshold,
                                               kind="gumbel")
                lf, sf, xf = fit_annual_maxima(annual_maxima, threshold,
                                               kind="frechet")
                lw, sw, xw = fit_annual_maxima(annual_maxima, threshold,
                                               kind="weibull")

            _P = np.linspace(P.min(), P.max(), num=1000)
            x, _x = map(_xaxis_transformation[xaxis], (P, _P))

            ax.plot(x, C, 'ok', label="Empirique", ms=2)
            ax.plot(x, lg-sg*np.log(-np.log(P)),
                    label="Gumbel  "
                    rf"$\mu={lg:.1f}$ $\sigma={sg:.1f}$ $\xi={xg:.2f}$")
            ax.plot(_x, poisson(_P, lf, sf, xf),
                    label="Fréchet "
                    rf"$\mu={lf:.1f}$ $\sigma={sf:.1f}$ $\xi={xf:+.2f}$")
            ax.plot(_x, poisson(_P, lw, sw, xw),
                    label=rf"Weibull "
                    rf"$\mu={lw:.1f}$ $\sigma={sw:.1f}$ $\xi={xw:+.2f}$")

            if xaxis == "probability":
                xt = ax.get_xticks()
                ax.set_xticks(xt)
                ax.set_xticklabels([f"{t:.0%}" for t in xt])
                ax.set_xlim(0, 1)
            ax.set_xlabel(_xaxis_label[xaxis])
            ax.set_ylabel("Quantiles des maxima\nannuels du débit (m$^3$/s)")
            ax.legend()
            if tight_layout:
                plt.tight_layout()

    return plt.show() if show else fig, ax


if __name__ == "__main__":
    t = 5
    rainfall = (0.00348136595716852, 0.00628034385649248, 0.011105325113109944, 0.01924833826669635, 0.03270162376850922, 0.05445772612213156, 0.08889222965104311, 0.1422270512638131, 0.22305641760337266, 0.34289515613651217, 0.5166805196955454, 0.7631272041636231, 1.1048056688589225, 1.5677938721795661, 2.1807513082025927, 2.973291101280017, 3.9735879576521436, 5.205258966386035, 6.683684813281432, 8.41208577236151, 10.37780565583667, 12.549357313885645, 14.87481410510137, 17.28206838152442, 19.681309995437026, 21.96981474580934, 24.038806312632108, 25.781816437883435, 27.10367843321912, 27.929108328328788, 28.209800533540324, 27.929108328328788, 27.10367843321912, 25.781816437883435, 24.038806312632108, 21.96981474580934, 19.681309995437026, 17.28206838152442, 14.87481410510137, 12.549357313885645, 10.37780565583667, 8.412085772361518, 6.683684813281429, 5.205258966386038, 3.9735879576521382, 2.973291101280017, 2.180751308202596, 1.5677938721795646, 1.1048056688589236, 0.7631272041636221, 0.5166805196955454, 0.3428951561365128, 0.22305641760337266, 0.1422270512638131, 0.08889222965104294, 0.05445772612213156, 0.03270162376850931, 0.01924833826669635, 0.011105325113109944, 0.00628034385649247, 0.00348136595716852, 0.0018916032183938975, 0.0010074523648748446, 0.0005259362523708125, 0.00026912609170705163, 0.000134987207161534, 6.636567713242961e-05, 3.198221644354147e-05, 1.5107329330452706e-05, 6.994890907281095e-06, 3.17459483183741e-06, 1.4122441179325348e-06, 6.158080397362672e-07, 2.63205551397356e-07, 1.1027037352850086e-07, 4.528316324762896e-08, 1.8227570148040377e-08, 7.191755432967916e-09, 2.781346856580493e-09, 1.0543614164118654e-09, 3.9177612625164015e-10, 1.4269230092385692e-10, 5.094214464944461e-11, 1.7826579433521266e-11, 6.114668582138957e-12, 2.0558524308654906e-12, 6.775245912399358e-13, 2.1886297501439247e-13, 6.93000653452448e-14, 2.1508449310701983e-14, 6.543327637416402e-15, 1.9512024998959755e-15, 5.703220082719332e-16, 1.6339999112484241e-16, 4.588788450616254e-17, 1.2631593902343577e-17, 3.408257378958827e-18, 9.014065773919483e-19, 2.33680876071006e-19, 5.937993692403821e-20, 1.479007587836416e-20, 3.610897683032292e-21, 8.641200328589346e-22, 2.026968850553373e-22, 4.66051796421864e-23, 1.0503533097852193e-23, 2.320335265142132e-24, 5.024353450344187e-25, 1.0664088396249394e-25, 2.2186122150691063e-26, 4.524318531589675e-27, 9.043552365954219e-28, 1.7718993636558216e-28, 3.402931108752661e-29, 6.405917957935036e-30, 1.1820169323363174e-30, 2.1378642935386854e-31, 3.7901000987567934e-32, 6.586206402328708e-33, 1.1218482215297003e-33, 1.8730396940726815e-34, 3.065306943306692e-35, 4.9171682310183656e-36, 7.731615989327032e-37, 1.1916249680234972e-37, 1.8002093937389474e-38, 2.665757029776473e-39, 3.8692983046711356e-40, 5.505008239883396e-41, 7.677111425721011e-42)
    analyse(rainfall, t, _base_functions=True)
