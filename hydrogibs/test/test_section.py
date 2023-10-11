import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    from hydrogibs import Section
else:
    from ..fluvial.canal import Section


for sheet, K, Js, h in zip(
    ('Rhone18.846', 'Rhone18.947'),
    (33, 32),
    (0.12/100, 0.13/100),
    (5, 5)
):
    df = pd.read_excel(
        'hydrogibs/test/section.xlsm',
        sheet_name=sheet,
        dtype=float
    )
    section = Section(
        df['Dist. cumulée [m]'],
        df['Altitude [m s.m.]'],
        K=K,
        i=Js
    )

    section.plot()
    plt.show()
