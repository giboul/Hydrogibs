from matplotlib import pyplot as plt
import numpy as np


class CircularCulvert:

    def __init__(self, D, J, K) -> None:
        self.diameter = D
        self.radius = D/2
        self.slope = J
        self.rugosity = K

    def get_theta(self, h):
        return np.arccos((self.radius-h)/self.radius)

    def get_height(self, theta):
        return self.radius * (1 - np.cos(theta))

    def get_area(self, h=None, theta=None):
        if h is None and theta is None:
            return ValueError("Provide h and/or theta")
        elif h is None:
            h = self.get_height(theta)
        elif theta is None:
            theta = self.get_theta(h)
        return self.radius**2 * theta - (self.radius-h) * np.sin(theta)

    def get_perimeter(self, h):
        return 2*self.get_theta(h) * self.radius

    def get_Rh(self, h):
        return self.get_area(h) / self.get_perimeter(h)

    def get_strickler_speed(self, h):
        S = self.get_area(h)
        Rh = self.get_Rh(h)
        # print(f"K = {self.rugosity}  {S = }  {Rh = }  {self.slope = :%}")
        return self.rugosity * Rh**(2/3) * np.sqrt(self.slope)

    def get_strickler_flow(self, h, *args, **kwargs):
        v = self.get_strickler_speed(h, *args, **kwargs)
        S = self.get_area(h)
        return S * v


def main():
    yticks = []
    for D in (1.2,):
        cul = CircularCulvert(D, 17/100, 1/0.011)
        q1 = cul.get_strickler_flow(D*0.4)
        q2 = cul.get_strickler_flow(D)
        l1 = plt.scatter(D, q1, c='r')
        l2 = plt.scatter(D, q2, c='g')
        yticks += [q1]
        # yticks += [q2]
        print(yticks)
    D = np.linspace(0.1, 1.6)
    Q = [None for _ in D]
    Qf = [None for _ in D]
    for i, d in enumerate(D):
        Qf[i] = CircularCulvert(d, 17/100, 1/0.011).get_strickler_flow(d)
        Q[i] = CircularCulvert(d, 17/100, 1/0.011).get_strickler_flow(d/2)
    plt.plot(D, Qf, 'g')
    plt.plot(D, Q, 'r')
    plt.ylim((-max(Q)*0.05, max(Q)*1.05))
    lq = plt.axline((1, 5), slope=0, ls='-.', color='k')
    plt.legend((l1, l2, lq), ('Remplisssage 50%', 'Remplisssage 100%', 'Débit imposé'))
    plt.xlabel('Diamètre (m)')
    plt.ylabel('Débit (m$^3$/s)')
    print(yticks)
    yticks += [5]
    plt.yticks(yticks)
    plt.title('Capacité de débit selon Manning-Strickler')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plt.style.use('seaborn')
    plt.figure(figsize=(5, 3))
    main()
