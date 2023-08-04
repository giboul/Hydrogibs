if __name__ == "__main__":
    from hydrogibs import Ressaut
else:
    from .. import Ressaut


def test(plot=False):
    r = Ressaut(
        q=10, i1=0.05, i2=0.002, p=0.5,
        h0=2, ms_K=50, x0=10, xt=20, dx=0.25
    )

    if plot:
        r.diagram(show=True, figsize=(5, 3))


if __name__ == "__main__":
    test(plot=True)
