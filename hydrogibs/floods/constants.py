class GR4preset:

    def __init__(self, S, R, X1, X2, X3, X4, Rt_ratio, group) -> None:
        self.S = S  # km²
        self.R = R  # Region
        self.Rt_ratio = Rt_ratio  # R/T %
        self.group = group
        self.X = (
            X1,  # [0:1]
            X2,  # mm
            X3,  # 1/mm
            X4,  # h
        )


_gr4_data = (
    ('Laval',        0.86, 'Alpes-du-Sud',    57.6, 7.28,  2.4, 0.38, 91, 1),
    ('Erlenbach',    0.64, 'Suisse Centrale', 46.5, 13.6, 16.2, 0.63, 53, 1),
    ('Rimbaud',       1.5, 'Alpes-du-Sud',    35.4,   40, 2.28, 1.07, 57, 2),
    ('Latte',        0.19, 'Massif Central',  14.4, 75.4, 3.96, 0.78, 41, 2),
    ('Sapine',       0.54, 'Massif Central',  15.7, 71.1, 0.90, 1.03, 34, 2),
    ('Rietholzbach', 3.31, 'Suisse Centrale', 26.5,   17, 2.82, 1.11, 41, 3),
    ('Lumpenenbach', 0.93, 'Suisse Centrale', 22.6, 12.2,  9.6,  0.5, 41, 3),
    ('Vogelbach',    1.55, 'Suisse Centrale', 31.4, 11.5, 5.88, 0.64, 56, 3),
    ('Brusquet',     1.08, 'Alpes-du-Sud',    13.8, 22.4, 0.72, 1.63, 54, 3)
)

GR4presets = {
    name: GR4preset(S, R, X1/100, X2, X3/100, X4, Rt_ratio, group)
    for name, S, R, X1, X2, X3, X4, Rt_ratio, group in _gr4_data
}


_qdf_data_mean = (
    ('Soyans',      0.87, 4.60,     0,  1.07, 2.50,  0.099, 0.569, 0.690, 0.046),
    ('Florac',      1.12, 3.56,     0,  0.95, 3.18,  0.039,  1.56,  1.91, 0.085),
    ('Vandenesse', 2.635, 6.19, 0.016, 1.045, 2.385, 0.172, 1.083,  1.75, 0)
)

_qdf_data_threshold = (
    ('Soyans',      2.57, 4.86,     0,  2.10,  2.10, 0.050,  1.49, 0.660, 0.017),
    ('Florac',      3.05, 3.53,     0,  2.13,  2.96, 0.010,  2.78,  1.77, 0.040),
    ('Vandenesse', 3.970, 6.48, 0.010, 1.910, 1.910, 0.097, 3.674, 1.774, 0.013)
)


def _arange_QDFcoefs(data):
    return {
        name: dict(A=alphas[:3], B=alphas[3:6], C=alphas[6:])
        for name, *alphas in data
    }


QDFcoefs_mean = _arange_QDFcoefs(_qdf_data_mean)
QDFcoefs_threshold = _arange_QDFcoefs(_qdf_data_threshold)
