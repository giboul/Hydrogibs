"""
hydrogibs

This package is meant to use analytic hydrology and fluid mechanics.
It does not make use of large programs of the finite element type.

The source material is publicly availabel and can be found on
Christophe Ancey's website https://en.ancey.ch/.
"""

from .floods import QDF, GR4, SoCoSe
from .floods.RationalMethod import rational_method
from .fluvial.canal import Section
from .extreme.yearly_maxima import YearlyMaxima
from .misc.misc import Ressaut, crupedix
