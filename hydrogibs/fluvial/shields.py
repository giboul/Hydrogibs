from scipy.interpolate import interp1d
import pandas as pd
from pathlib import Path
if __name__ == "__main__":
    from hydrogibs import constants as cst
else:
    from .. import constants as cst

rho_s = 250
rho = cst.rho
g = cst.g
nu_k = 1.316e-06


def adimensional_diameter(di, solid_density, nu=nu_k):
    return di*((solid_density/rho-1)*g/nu_k**2)**(1/3)


def Reynolds(u_star, d, nu=nu_k):
    return u_star * d / nu


def adimensional_shear(shear, d, solid_density, g=g):
    return shear/((solid_density - rho)*g*d)


shields = pd.read_csv(Path(__file__).parent / "shields.csv")


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    grains = pd.read_csv(f"hydrogibs/test/fluvial/grains.csv")
    granulometry = interp1d(grains["Tamisats [%]"],
                            grains["Diamètre des grains [cm]"])
    d16, d50, d90 = granulometry((16, 50, 90))
    print(shields)
    plt.semilogx(shields.reynolds, shields.shear)
    plt.show()
