if __name__ == "__main__":
    from hydrogibs.floods import QDF
else:
    from ..floods import QDF
from matplotlib import pyplot as plt


def test(plot=False):
    # Estimation of Q(T=100) according to the QdF method
    catchment = QDF.Catchment(model="soyans", specific_duration=1)
    rain = QDF.Rain(
        duration=1,
        return_period=100,
        specific_discharge=1.3,
        discharge_Q10=1.3,
        dt=0.01
    )

    # QDF.App(rain=rain, catchment=catchment, style="ggplot")
    event = rain @ catchment

    if plot:
        plt.plot(event.time, event.discharge)
        plt.xlabel("Time [h]")
        plt.ylabel("Discharge [m$^3$/s]")
        plt.show()


if __name__ == "__main__":
    test(plot=True)
