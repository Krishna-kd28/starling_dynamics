"""Simulated strapdown IMU (gyro + accelerometer) and an optional tilt
estimate, read from a :class:`~starling_dynamics.dynamics.QuadState`.

::

    gyro  = w_body + bias_g + noise                              [rad/s]
    accel = specific force = R^T (a_world - g) + bias_a + noise   [m/s^2]

with world ``z`` down and gravity ``g = [0, 0, G]``. In FRD body axes a level
hover therefore reads ``accel ~ [0, 0, -G]`` (the upward thrust reaction) and
free fall reads ``accel ~ 0``.

Biases are per-episode constants (redrawn on reset); noise is white per read.
Default magnitudes are for the TDK InvenSense ICM-42688-P (the Starling 2 IMU);
see the package README. The optional ``tilt`` output models an IMU-only
roll/pitch estimate (observable without external aiding; yaw is not): true
roll/pitch plus a slow random-walk drift plus noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dynamics import G, QuadState
from .quaternion import euler_zyx_from_quat, quat_rotate_inv


@dataclass
class ImuParams:
    gyro_bias: torch.Tensor    # (B,3) rad/s, per-episode
    accel_bias: torch.Tensor   # (B,3) m/s^2, per-episode
    gyro_noise: float = 0.005  # rad/s per read
    accel_noise: float = 0.10  # m/s^2 per read
    tilt_noise: float = 0.005  # rad per read
    tilt_drift_rate: float = 0.002  # rad/sqrt(s) random walk

    @staticmethod
    def nominal(batch: int, device="cpu") -> "ImuParams":
        """Zero-bias parameters (noise defaults unchanged)."""
        z = torch.zeros((batch, 3), device=device)
        return ImuParams(gyro_bias=z, accel_bias=z.clone())

    @staticmethod
    def randomized(batch: int, device="cpu", g: torch.Generator | None = None,
                   gyro_bias_sigma: float = 0.02, accel_bias_sigma: float = 0.2) -> "ImuParams":
        rn = lambda s: torch.randn((batch, 3), device=device, generator=g) * s
        return ImuParams(gyro_bias=rn(gyro_bias_sigma), accel_bias=rn(accel_bias_sigma))


class ImuSim:
    """Simulated IMU. Stateful only for the tilt-drift random walk."""

    def __init__(self, params: ImuParams, dt: float):
        self.p = params
        self.dt = dt
        self.tilt_drift = torch.zeros_like(params.gyro_bias[:, :2])

    def reset(self, params: ImuParams | None = None):
        if params is not None:
            self.p = params
        self.tilt_drift = torch.zeros_like(self.p.gyro_bias[:, :2])

    @torch.no_grad()
    def read(self, s: QuadState, g: torch.Generator | None = None):
        """Return a dict of sensor readings ``{"gyro", "accel", "tilt"}``."""
        dev = s.p.device
        rnd = lambda shape, sig: torch.randn(shape, device=dev, generator=g) * sig
        gyro = s.w.detach() + self.p.gyro_bias + rnd(s.w.shape, self.p.gyro_noise)
        g_vec = torch.tensor([0.0, 0.0, G], device=dev).expand_as(s.v)
        a_w = s.a_world.detach() if s.a_world is not None else torch.zeros_like(s.v)
        accel = quat_rotate_inv(s.q.detach(), a_w - g_vec) \
            + self.p.accel_bias + rnd(s.v.shape, self.p.accel_noise)
        _, pitch, roll = euler_zyx_from_quat(s.q.detach())
        self.tilt_drift = self.tilt_drift + rnd(self.tilt_drift.shape,
                                                self.p.tilt_drift_rate) * (self.dt ** 0.5)
        tilt = torch.stack((roll, pitch), dim=-1) + self.tilt_drift \
            + rnd(self.tilt_drift.shape, self.p.tilt_noise)
        return {"gyro": gyro, "accel": accel, "tilt": tilt}
