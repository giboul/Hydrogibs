import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path


x = np.array((0, 1, 2))
z = np.array((2, 0, 5))

h = np.linspace(0, z[0])
w = np.linspace(0, x.max())
P = np.sqrt(np.diff(x)**2 + np.diff(z**2)).sum()
S = w*h/2

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(x, z)
ax2.plot(h, S, label="S")
ax2.plot(h, P, label="P")
ax2.plot(h, S*(S/P)**(2/3), label="Q")
plt.legend()
plt.show()

pd.DataFrame(
    np.hstack((x, z)).T, columns=("x", "z")
).to_csv(Path(__file__).parent/"minimalProfile.csv", index=False)
