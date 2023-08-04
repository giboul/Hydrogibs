import numpy as np
if __name__ == "__main__":
    from hydrogibs import rational_method
else:
    from .. import rational_method
from matplotlib import pyplot as plt


def test(plot=False):
    durations = np.array((0.5, 1, 1.5, 2, 2.5))

    def rainfall(d, T):
        return 51.0 - 88.2 * (1-(-np.log(1-1/T))**-0.107) * (d/24)**0.4

    for d in durations:
        t, Q = rational_method(S=1.8, Cr=0.8, tc=d, ip=rainfall(d, 100)/d)
        if plot:
            plt.plot(t, Q, label=f"{d:.1f} h")
    if plot:
        plt.xlabel("Temps (h)")
        plt.ylabel("Débit (m$^3$/s)")
        plt.legend(title="Durée (h)")
        plt.show()


if __name__ == "__main__":
    test(plot=True)
