import numpy as np


class SoCoSe:

    def __init__(self,
                 S: float,
                 L: float,
                 Pa: float,
                 P10: float,
                 ta: float,
                 Montana2: float,
                 zeta: float = 1.0,
                 tf: float = 5,
                 dt: float = 0.01) -> None:

        self.Pa = Pa
        self.P10 = P10

        self.ds = -0.69 + 0.32*np.log(S) + 2.2*np.sqrt(Pa/(P10*ta))
        self.J = 260 + 21*np.log(S/L) - 54*np.sqrt(Pa/P10)

        k = 24**Montana2/21*P10/(1 + np.sqrt(S)/(30*self.ds**(1/3)))
        rho = 1 - 0.2 * self.J / (k * (1.25*self.ds)**(1-Montana2))
        self.Q10 = zeta * k*S * rho**2 / ((15-12*rho)*(1.25*self.ds)**Montana2)

        self.time = np.arange(0, tf, step=dt)
        tau = 2*self.time/(3*self.ds)
        self.Q = self.Q10 * 2 * tau**4 / (1 + tau**8)
