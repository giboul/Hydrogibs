from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
if __name__ == "__main__":
    from hydrogibs import constants as cst
else:
    from .. import constants as cst

rho_s = 2650
rho = cst.rho
g = cst.g
nu_k = cst.nu_k


def adimensional_diameter(di, solid_density, nu=nu_k):
    return di*((solid_density/rho-1)*g/nu_k**2)**(1/3)


def reynolds(u_star, d, nu=nu_k):
    return u_star * d / nu


def adimensional_shear(shear, d, solid_density, g=g):
    return shear/((solid_density - rho)*g*d)


def smart_jaeggi(h, i, s, theta_cr, Dm):
    return 4.2/(s - 1) * i**1.6 * (1 - theta_cr*(s-1)*Dm/h/i)


def shields_diagram(diameters, hydraulic_radii, slope,
                    diameter_labels=None,
                    show=True,
                    fig=None,
                    axes=None,
                    **plot_kwargs):

    Rh = np.asarray(hydraulic_radii)

    if fig is None:
        fig = plt.gcf()
    if axes is None:
        axes = fig.subplots(ncols=2, gridspec_kw=dict(wspace=0))
    ax1, ax2 = axes

    # ax1.set_prop_cycle(None)
    # ax2.set_prop_cycle(None)

    for d, dlab in zip(diameters, diameter_labels):
        shear = rho*g*Rh*slope
        r = reynolds(np.sqrt(shear/rho), d)
        s = adimensional_shear(shear, d, rho_s)
        d = adimensional_diameter(d, rho_s)
        ax1.plot(r, s, label=dlab, **plot_kwargs)
        ax2.plot(np.full_like(s, d), s, **plot_kwargs)

    # ax1.set_prop_cycle(None)
    # ax2.set_prop_cycle(None)

    ax1.plot(shields.reynolds, shields.shear)
    ax2.plot(vanrijn.diameter, vanrijn.shear)

    ax1.loglog()
    ax2.loglog()

    ax1.set_xlabel(r"Reynolds $R=u_\ast d/\nu$")
    ax1.set_ylabel("Cisaillement critique adimensionnel\n"r"$\Theta=\tau/(\rho g[s-1]d)$")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_xlabel(r"Diamètre adimentionnel $d_\ast=d\cdot\sqrt[3]{(s-1)g/\nu^2}$")
    ax2.set_ylabel("Cisaillement critique adimensionnel\n"r"$\Theta=\tau/(\rho g[s-1]d)$")
    ax1.set_title("Diagramme de Shields")
    ax2.set_title("Selon Van Rijn")

    if diameter_labels is not None:
        ax1.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


DIR = Path(__file__).parent
shields = pd.read_csv(DIR / "shields.csv")
vanrijn = pd.read_csv(DIR / "shields-vanrijn.csv")


if __name__ == "__main__":

    hdata = pd.read_csv("hydrogibs/fluvial/hydraulic_data.csv").query("300 <= Q <= 1600")

    grains = pd.read_csv("hydrogibs/fluvial/grains.csv")
    granulometry = interp1d(grains["Tamisats [%]"],
                            grains["Diamètre des grains [cm]"])
    d16, d50, d90 = granulometry((16, 50, 90))
    shields_diagram((d16/100, d50/100, d90/100),
                    hdata.S/hdata.P, lope=0.12/100,
                    diameter_labels=("$d_{16}$", "$d_{50}$", "$d_{90}$", ))
