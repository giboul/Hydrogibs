from hydrogibs import Ressaut
from hydrogibs import crupedix
import hydrogibs as hg
import numpy as np
from scipy.stats import skewnorm
from matplotlib import pyplot as plt
plt.style.use("ggplot")


Q10 = crupedix(S=1.8, Pj10=72, R=1.0)

# DEMO QDF method
# Estimation of Q(T=100) according to the QdF method
catchment = hg.QDF.Catchment(model="soyans", specific_duration=1)
rain = hg.QDF.Rain(duration=1,
                   return_period=100,
                   specific_discharge=Q10,
                   discharge_Q10=Q10,
                   dt=0.01)

event = rain @ catchment

plt.plot(event.time, event.discharge)
plt.title("QDF method")
plt.xlabel("Time [h]")
plt.ylabel("Discharge [m$^3$/s]")
plt.show()

# DEMO GR4 method
I0 = 66.7  # mm/h
dt = 0.01
time = np.arange(0, 24, step=dt)
unit_rain = np.exp(-(time - 3)**2)
unit_rain = skewnorm.pdf(time, 3, loc=3, scale=1)
unit_rain = unit_rain / np.trapz(x=time, y=unit_rain)
rain = hg.GR4.Rain(time, unit_rain * I0)

catchment = hg.GR4.PresetCatchment("Rimbaud", surface=1.8)

hg.GR4.App(catchment, rain)

# event = rain @ catchment
# Qax, Pax, Vax = event.diagram(show=False).axes
# Pax.set_title("Rimbaud")
# plt.show()

# DEMO rational_method
durations = np.array((0.5, 1, 1.5, 2, 2.5))


def rainfall(d, T):
    return 51.0 - 88.2 * (1-(-np.log(1-1/T))**-0.107) * (d/24)**0.4


for d in durations:
    t, Q = hg.rational_method(S=1.8, Cr=0.8, tc=d, ip=rainfall(d, 100)/d)
    plt.plot(t, Q, label=f"{d:.1f} h")
plt.title("Rational Method")
plt.xlabel("Temps (h)")
plt.ylabel("Débit (m$^3$/s)")
plt.legend(title="Durée (h)")
plt.show()

# DEMO Ressaut
Ressaut(
    q=10, i1=0.05, i2=0.002, p=0.5,
    h0=2, ms_K=50, x0=10, xt=20, dx=0.25
).diagram(show=True, figsize=(10, 3))
