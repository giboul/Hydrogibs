import numpy as np


def rational_method(S: float,
                    Cr: float,
                    tc: float,
                    ip: float = 1.0,
                    dt: float = 0.01) -> tuple:
    """
    Computes a triangular hydrogram from a flood with volume Cr*tc*S

    Args:
        S (float): Catchemnt area
        Cr (float): Peak runoff coefficient
        tc (float): Concentration time
        ip (float) [mm/h]: Rainfall intensity
        dt (float): timestep, default to 1 if not specified

    Returns:
        time [h], discharge [m^3/s] (numpy.ndarray, numpy.ndarray)
    """

    q = Cr*ip*S
    Qp = q/3.6

    time = np.arange(0, 2*tc, step=dt)
    Q = np.array([
        Qp * t/tc if t < tc else Qp * (2 - t/tc)
        for t in time
    ])

    return time, Q


def main():
    from matplotlib import pyplot as plt
    t, Q = rational_method(S=1, Cr=0.6, tc=1)
    plt.plot(t, Q)
    plt.xlabel("Time [h]")
    plt.ylabel("Flow [m$^3$/s]")
    plt.show()


if __name__ == "__main__":
    main()
