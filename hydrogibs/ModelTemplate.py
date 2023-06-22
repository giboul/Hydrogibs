class Rain:

    def __init__(self, *args, **kwargs) -> None:
        return NotImplemented

    def __matmul__(self, catchment):
        return model(rain=self, catchment=catchment)

    def __repr__(self) -> str:
        s = "\n".join(
            f"\t{k}: {v}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}: {'{'}\n{s}\n{'}'}"


class Catchment:

    def __init__(self, *args, **kwargs) -> None:
        return NotImplemented

    def __matmul__(self, rain):
        return rain @ self

    def __repr__(self) -> str:
        s = "\n".join(
            f"\t{k}: {v}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}: {'{'}\n{s}\n{'}'}"


class Event:

    def __init__(self, rain: Rain, catchment: Catchment) -> None:
        return NotImplemented

    def diagram(self, *args, **kwargs):
        return Diagram(self, *args, **kwargs)

    def __repr__(self) -> str:
        s = "\n".join(
            f"\t{k}: {v.max()}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}: {'{'}\n{s}\n{'}'}"

    def __format__(self, __format_spec: str) -> str:
        s = "\n".join(
            f"\t{k}: {v.max():{__format_spec}}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__} maximums: {'{'}\n{s}\n{'}'}"


# class Model:

#     def __init__(self, *args, **kwargs) -> None:
#         return NotImplemented

#     def diagram(self, *args, **kwargs):
#         return NotImplemented

#     def App(self, *args, **kwargs):
#         return NotImplemented


def model(catchment: Catchment, rain: Rain) -> Event:
    return NotImplemented


class Diagram:

    def __init__(self, *args, **kwargs) -> None:
        return NotImplemented

    def update(self, canvas=None):
        return NotImplemented

    def zoom(self):
        return NotImplemented
