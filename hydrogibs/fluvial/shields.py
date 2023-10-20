import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd


constants = pd.read_csv(f"hydrogibs/test/fluvial/constants.csv")
rho_s, rho, g, nu = constants.Valeur.to_numpy().T


def adimensionate_diameter(di, solid_density):
    return di*((solid_density/rho-1)*g/nu**2)**(1/3)


def adimensional_shear(shear, d, solid_density):
    return shear/((solid_density - rho)*g*d)


if __name__ == "__main__":

    grains = pd.read_csv(f"hydrogibs/test/fluvial/grains.csv")
    granulometry = interp1d(grains["Tamisats [%]"],
                            grains["Diamètre des grains [cm]"])
    d16, d50, d90 = granulometry((16, 50, 90))

    theta = np.linspace(0, grains["Diamètre des grains [cm]"].max())
