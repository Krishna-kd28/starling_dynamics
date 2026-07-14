"""starling_dynamics: batched quadrotor dynamics + IMU (PyTorch).

A self-contained, framework-agnostic simulation of a small quadrotor (defaults
for the ModalAI Starling 2) with two command interfaces (CTBR and attitude
setpoint) and a strapdown IMU.

See the package README for the model, units/frames, parameters, and usage.
"""

from .dynamics import (ATT_RATE_LIMIT, DynParams, G, KATT_DR, KATT_RP_NOM,
                       KATT_Y_NOM, KD_LIN_FLOOR, QuadAttitudeDynamics,
                       QuadCTBRDynamics, QuadState, TILT_SP_LIMIT)
from .imu import ImuParams, ImuSim
from .quaternion import (euler_zyx_from_quat, quat_conj, quat_exp_map,
                         quat_from_euler_zyx, quat_mul, quat_normalize,
                         quat_rotate, quat_rotate_inv)

__version__ = "1.0.0"

__all__ = [
    "G", "KD_LIN_FLOOR", "TILT_SP_LIMIT", "KATT_RP_NOM", "KATT_Y_NOM",
    "ATT_RATE_LIMIT", "KATT_DR",
    "DynParams", "QuadState", "QuadCTBRDynamics", "QuadAttitudeDynamics",
    "ImuParams", "ImuSim",
    "quat_mul", "quat_conj", "quat_rotate", "quat_rotate_inv", "quat_normalize",
    "quat_exp_map", "quat_from_euler_zyx", "euler_zyx_from_quat",
]
