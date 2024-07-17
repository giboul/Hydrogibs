from pathlib import Path
from pandas import DataFrame
from matplotlib import pyplot as plt


x = [0, 1, 2, 3, 4, 5, 4, 3, 6, 6, 9]
y = [9, 0, 3, 3, 0, 0, 1, 4, 0, 0, 9]

plt.plot(x, y)
plt.show()

DataFrame(
    data=list(zip(x, y)),
    columns=["x", "z"]
).to_csv(Path(__file__).parent/"randomProfile.csv")
