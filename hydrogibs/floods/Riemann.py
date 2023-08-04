import numpy as np
from matplotlib import pyplot as plt
from ..constants import g, rho, mu


def rankine_hugoniot(h, h0, i, K):
    return K * np.sqrt(i) * (h**(5/3) - h0**(5/3))/(h - h0)


def vague(x, t, h0, length, theta, rho=rho, g=g, mu=mu):
    hmax = np.sqrt(mu/(rho * g * np.sin(theta)))

    h = np.full_like(x, hmax)
    A = (9/4 * rho*g*h0**2*length**2*np.sin(theta)/mu)**(1/3)
    xf = A * t**(1/3)
    wave_slice = 0 <= x < xf
    h[wave_slice] = np.sqrt(mu*x/(rho*g*np.sin(theta)*t))


class Riemann:

    _colors = ('k', 'b', 'r')

    def __init__(self, x, h, theta):
        self.position = x
        self.height = h
        self.theta = theta
        self.calculate()

    def precalculate(self, rho=rho, g=g, mu=mu, slice=1):
        self.csign = np.sign(np.diff(
            self.height,
            prepend=self.height[0]
        )).astype(int)
        self.c = rho * g * self.height[::slice]**2 * np.sin(self.theta) / mu
        self.time = self.position**2
        return self

    def calculate(self):
        self.precalculate()
        return self

    def diagram(self, dx=1, style='bmh', show=True, *figargs, **figkwargs):
        with plt.style.context(style):
            fig, (axt, axb) = plt.subplots(
                nrows=2,
                sharex=True,
                gridspec_kw=dict(
                    hspace=0.0,
                    height_ratios=[1, 1]
                ),
                *figargs, **figkwargs
            )
            axb.plot(self.position, self.height, c='r')
            slice = self.position.size // self.c.size
            for x, c, sign in zip(self.position[::slice], self.c, self.csign):
                axt.plot((x, x+dx), (0, dx/c),
                         ls='-', c=self._colors[sign], lw=1)

            axt.set_ylabel("t (s)")
            axb.set_ylabel('h (m)')
            axb.set_xlabel('x (m)')
            axt.set_ylim((0, (dx/self.c).min()))

            if show:
                plt.show()
        return fig


if __name__ == "__main__":
    hmax = 1
    tf = 50
    _t = np.linspace(0, tf)
    h = np.full_like(_t, hmax)
    h[_t < 0.2*tf] = 0.5 * hmax
    h[_t > 0.8*tf] = 0.3 * hmax
    Riemann(_t, h, 2 * np.pi/180).calculate().diagram()
