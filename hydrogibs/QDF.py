import numpy as np
from hydrogibs.misc import crupedix, Turraza


class QdF:

    """
    Based on rainfall GradEx,
    can estimate discharges for catchments of model type:
        - Soyans
        - Florac
        - Vandenesse

    Args:
        model (str):          Either 'Soyans', 'Florac' or 'Vandenesse'
        ds    (float) [h]:    Specific duration
        S     (float) [km^2]: Catchment surface
        L     (float) [km]:   Length of the thalweg
        im    (float) [%]:    Mean slope of the thalweg

    Calculates:
        tc (float) [h]: concentration time
    """

    _coefs = dict(

        soyans=dict(
            A=(2.57, 4.86, 0),
            B=(2.10, 2.10, 0.050),
            C=(1.49, 0.660, 0.017)),

        florac=dict(
            A=(3.05, 3.53, 0),
            B=(2.13, 2.96, 0.010),
            C=(2.78, 1.77, 0.040)),

        vandenesse=dict(
            A=(3.970, 6.48, 0.010),
            B=(1.910, 1.910, 0.097),
            C=(3.674, 1.774, 0.013))
    )

    def __init__(self, model, ds, S, L, im) -> None:
        """
        Based on rainfall GradEx,
        can estimate discharges for catchments of model type:
            - Soyans
            - Florac
            - Vandenesse

        Args:
            model (str):          Either 'Soyans', 'Florac' or 'Vandenesse'
            ds    (float) [h]:    Specific duration
            S     (float) [km^2]: Catchment surface
            L     (float) [km]:   Length of the thalweg
            im    (float) [%]:    Mean slope of the thalweg

        Calculates:
            tc (float) [h]: concentration time
        """
        self.coefs = self._coefs[model]
        self.ds = ds
        self.im = im
        self.S = S
        self.L = L

        self.tc = Turraza(S, L, im)

    def _calc_coefs(self, a):
        a1, a2, a3 = a
        return 1/(a1*self._d/self.ds + a2) + a3

    def discharge(self, d, T, Qsp, Q10):
        """
        Estimates the discharge for a certain flood duration
        and a certain return period

        Args:
            d (numpy.ndarray) [h]: duration of the flood
            T (numpy.ndarray) [y]: Return period
            Qsp (numpy.ndarray): Specific discharge
            Q10 (numpy.ndarray): Discharge for return period of 10 years

        Returns:
            (numpy.ndarray): Flood discharge
        """

        self._d = np.asarray(d)
        Qsp = np.asarray(Qsp)
        Q10 = np.asarray(Q10)
        T = np.asarray(T)

        self.A, self.B, self.C = map(
            self._calc_coefs,
            self.coefs.values()
        )
        return Q10 + Qsp * self.C * np.log(1 + self.A * (T-10)/(10*self.C))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    qdf = QdF(model="soyans", ds=1, S=1.8, L=2, im=25)
    Q10 = crupedix(S=1.8, Pj10=72, R=1.75)
    d = np.linspace(0, 3)
    Q100 = qdf.discharge(d, 100, Q10, Q10)
    plt.plot(d, Q100)
    plt.show()
