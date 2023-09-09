import numpy as np
import pandas as pd
from os.path import join, dirname
from matplotlib import pyplot as plt
if __name__ == "__main__":
    from hydrogibs import GR4
else:
    from ..floods import GR4


def test(plot=False, app=False):
    # Custom test
    X1 = 57.6/100  # [-] dR = X1*dP
    X2 = 7.28  # [mm] Interception par la végétation
    X3 = 2.4/100  # [h^-1] dH = X3*V*dt, V = (1-X1)*I*dt
    X4 = 0.38  # [h] temps de montée tm ≃ td

    t0 = 1  # h
    I0 = 66.7  # mm/h

    # df = pd.read_csv(join(dirname(__file__), "rain.csv"))
    # time = pd.to_datetime(df.Date, format="%Y-%m-%d %H:%M:%S")
    # rainfall = df.Rainfall
    dt = 0.01
    time = np.arange(0, 24, step=dt)
    rainfall = np.exp(-(time - 3)**2)
    rainfall = I0 * rainfall / np.trapz(x=time, y=rainfall)

    rain = GR4.Rain(time, rainfall)
    catchment = GR4.Catchment(X1, X2, X3, X4, surface=1.8)

    event = rain @ catchment

    if plot:
        Qax, Pax, Vax = event.diagram(show=False).axes
        Pax.set_title("Rimbaud")
        plt.show()

    # rain = GR4.BlockRain(I0, t0)
    catchment = GR4.PresetCatchment('Laval')
    event = rain @ catchment

    if app:
        GR4.App(catchment, rain)


if __name__ == "__main__":
    test(app=True)
