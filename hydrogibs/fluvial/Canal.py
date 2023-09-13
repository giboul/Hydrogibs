from typing import Callable
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Canal:

    surface: Callable
    perimeter: Callable
    friction_law: Callable
    slope: float | np.ndarray
    friction_law: Callable


class TrapCanal(Canal):

    def __init__():
