import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path


r = 1
alpha = np.hstack((
    np.linspace(0, np.pi/2, num=10),
    np.linspace(np.pi/2, np.pi, num=20, endpoint=True)
))
theta = 2*alpha + np.pi/2

x = r * np.cos(theta)
z = r * np.sin(theta)
print()
x[np.isclose(x, 0.)] = 0.

S = (np.pi-alpha + np.sin(alpha)/2)*r**2
P = 2*(np.pi-alpha)*r

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(x, z)
ax2.plot(alpha, S, label="S")
ax2.plot(alpha, P, label="P")
ax2.plot(alpha, S*(S/P)**(2/3), label="Q")
plt.legend()
plt.show()

pd.DataFrame(
    data = np.vstack((x, z)).T, columns=("x", "z")
).to_csv(Path(__file__).parent/"closedProfile.csv", index=False)
