class Rain:

    def __init__(self, *args, **kwargs) -> None:
        return NotImplemented

    def __matmul__(self, catchment):
        return model(rain=self, catchment=catchment)


class Catchment:

    def __init__(self, *args, **kwargs) -> None:
        return NotImplemented

    def __matmul__(self, rain):
        return rain @ self


class Event:

    def __init__(self, rain: Rain, catchment: Catchment) -> None:
        return NotImplemented

    def diagram(self, *args, **kwargs):
        return Diagram(self, *args, **kwargs)


class Diagram:

    def __init__(self, *args, **kwargs) -> None:
        return NotImplemented

    def update(self, canvas=None):
        return NotImplemented

    def zoom(self):
        return NotImplemented


# class Model:

#     def __init__(self, *args, **kwargs) -> None:
#         return NotImplemented

#     def diagram(self, *args, **kwargs):
#         return NotImplemented

#     def App(self, *args, **kwargs):
#         return NotImplemented


def model(catchment: Catchment, rain: Rain) -> Event:
    return NotImplemented
