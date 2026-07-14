"""Batched quaternion utilities (PyTorch).

Convention
----------
- Quaternions are stored ``(w, x, y, z)`` (scalar-first) in the last dim.
- A quaternion ``q`` represents a **body->world** rotation: ``quat_rotate(q, v)``
  maps a body-frame vector to world, ``quat_rotate_inv(q, v)`` maps world->body.
- Hamilton product (right-handed), consistent with SciPy's
  ``Rotation`` and the ZYX Euler helpers below.
- All functions are batched over leading dims.

Euler angles use the intrinsic Z-Y-X (yaw, pitch, roll) sequence, matching
``scipy.spatial.transform.Rotation.from_euler('ZYX', (yaw, pitch, roll))``.
"""

from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b``. Shapes ``(...,4) x (...,4) -> (...,4)``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate ``(w, -x, -y, -z)``; equals the inverse for a unit quaternion."""
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a body-frame vector into world: ``R(q) @ v``.

    Shapes ``(...,4), (...,3) -> (...,3)``.
    """
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame vector into body: ``R(q)^T @ v``.

    Shapes ``(...,4), (...,3) -> (...,3)``.
    """
    qw = q[..., :1]
    qv = -q[..., 1:]
    t = 2.0 * torch.linalg.cross(qv, v, dim=-1)
    return v + qw * t + torch.linalg.cross(qv, t, dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    """Return ``q`` scaled to unit norm (safe near zero)."""
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_exp_map(w: torch.Tensor, dt) -> torch.Tensor:
    """Quaternion increment ``exp(0.5 * w * dt)`` for a body angular rate ``w``.

    Given a (constant over ``dt``) body rate ``w`` (rad/s), returns the unit
    quaternion of the rotation ``|w|*dt`` about axis ``w/|w|``. Use as
    ``q_next = quat_normalize(quat_mul(q, quat_exp_map(w, dt)))``.

    Shapes ``(...,3) -> (...,4)``. ``dt`` may be a scalar or a ``(...)`` tensor.
    Uses a sinc expansion for numerical stability as ``|w| -> 0``.
    """
    dt_ = dt if not torch.is_tensor(dt) else dt.unsqueeze(-1)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True) * dt_ * 0.5
    half = w * dt_ * 0.5
    small = theta < 1e-8
    k = torch.where(small, 1.0 - theta * theta / 6.0,
                    torch.sin(theta) / theta.clamp_min(1e-12))
    return torch.cat((torch.cos(theta), k * half), dim=-1)


def quat_from_euler_zyx(yaw, pitch, roll) -> torch.Tensor:
    """Intrinsic Z-Y-X Euler (yaw, pitch, roll) -> quaternion ``(w,x,y,z)``.

    Matches ``scipy Rotation.from_euler('ZYX', (yaw, pitch, roll))``.
    Inputs broadcast; output has a trailing dim of 4.
    """
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    return torch.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ),
        dim=-1,
    )


def euler_zyx_from_quat(q: torch.Tensor):
    """Quaternion -> ``(yaw, pitch, roll)`` intrinsic Z-Y-X. Inverse of
    :func:`quat_from_euler_zyx`. Pitch is clamped to ``[-pi/2, pi/2]``
    (gimbal-lock guard). Returns three tensors of shape ``q.shape[:-1]``.
    """
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    sinp = (2 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll
