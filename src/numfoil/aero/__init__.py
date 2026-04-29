# from .m_constant_vortex import ConstantVortex
# from .m_lumped_vortex import LumpedVortex

from numfoil.aero.neuralfoil.neuralfoil_wrapper import NeuralFoil
from numfoil.aero.xfoil.xfoil import XFoil

__all__ = ["XFoil", "NeuralFoil"]
