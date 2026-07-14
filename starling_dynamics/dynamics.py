"""Batched quadrotor rigid-body dynamics (PyTorch).

Two command interfaces onto the same plant:

- :class:`QuadCTBRDynamics` — collective-thrust + body-rate (CTBR) action.
- :class:`QuadAttitudeDynamics` — attitude-setpoint + thrust action, adding a
  PX4-style quaternion-P attitude cascade on top of the same rate loop.

The autopilot inner loop (rate controller + motors) is abstracted as
first-order tracking of the commanded body rates / collective thrust, plus a
transport-delay FIFO on the command, linear rotor drag, and per-episode
physical-parameter randomization. Default parameters are for the ModalAI
Starling 2 (see the package README for values and sources).

Frames & units
--------------
- World frame, metres. ``z`` points **DOWN** (NED-like); gravity ``= +G z``.
- Body frame, **FRD** (x forward, y right, z down). Collective thrust acts
  along ``-z_body``.
- Attitude quaternion ``q`` maps **body -> world**, stored ``(w, x, y, z)``.

State (all torch, batch ``B``)
------------------------------
- ``p`` (B,3) position [m], ``v`` (B,3) velocity [m/s] (world frame),
- ``q`` (B,4) unit quaternion body->world,
- ``w`` (B,3) body angular rates [rad/s],
- ``thrust`` (B,) produced mass-normalized collective thrust [m/s^2],
- ``cmd_fifo`` (B, D, 4) queue of pending (delayed) commands,
- ``a_world`` (B,3) last linear acceleration [m/s^2] (world frame; for the IMU).

Action (B,4)
------------
- CTBR: ``[c, wx_cmd, wy_cmd, wz_cmd]`` — ``c`` mass-normalized collective
  thrust [m/s^2] clamped to ``[0, twr*G]``; body-rate setpoints [rad/s]
  clamped to :attr:`QuadCTBRDynamics.RATE_LIMIT`.
- Attitude: ``[c, roll_sp, pitch_sp, yaw_sp]`` — ``c`` as above; roll/pitch
  setpoints [rad] clamped to :data:`TILT_SP_LIMIT`; ``yaw_sp`` is an absolute
  yaw setpoint [rad] (the cascade wraps the error, shortest path).

Every physical parameter is a ``(B,)``-broadcastable tensor, so domain
randomization is just sampling :class:`DynParams` per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .quaternion import (euler_zyx_from_quat, quat_conj, quat_exp_map,
                         quat_from_euler_zyx, quat_mul, quat_normalize,
                         quat_rotate)

G = 9.81  # gravitational acceleration [m/s^2]

# Lower bound of the per-episode linear rotor-drag range (see DynParams).
KD_LIN_FLOOR = 0.03

# ---- attitude-setpoint interface constants (QuadAttitudeDynamics) ----
TILT_SP_LIMIT = 0.35   # rad: clamp on commanded roll/pitch setpoints
KATT_RP_NOM = 16.0     # attitude-P roll/pitch gain [1/s] (Starling 2 D0014
                       # MC_ROLL_P / MC_PITCH_P)
KATT_Y_NOM = 2.8       # attitude-P yaw gain [1/s] (Starling 2 D0014 MC_YAW_P)
# The attitude controller clamps its rate output at MC_*RATE_MAX =
# 130/130/150 deg/s on the Starling 2 (D0014). This applies only inside the
# attitude cascade; CTBR rate setpoints use QuadCTBRDynamics.RATE_LIMIT.
ATT_RATE_LIMIT = (2.269, 2.269, 2.618)  # rad/s
KATT_DR = 0.20         # +/-20% randomization on the attitude-cascade gains


# ------------------------------------------------------------------- params
@dataclass
class DynParams:
    """Per-episode physical parameters, each a ``(B,)`` tensor.

    ``katt_rp`` / ``katt_y`` are used only by :class:`QuadAttitudeDynamics`;
    they are ``None`` in CTBR mode and the attitude plant falls back to the
    nominal gains when unset.
    """

    twr: torch.Tensor          # thrust-to-weight ratio; max thrust = twr*G [m/s^2]
    tau_w: torch.Tensor        # rate-loop closed-loop time constant [s]
    tau_c: torch.Tensor        # thrust (motor) time constant [s]
    kd_lin: torch.Tensor       # linear rotor-drag coefficient [1/s]
    delay_steps: torch.Tensor  # transport delay in control steps (B,) long
    thrust_gain: torch.Tensor  # multiplicative thrust-map error (battery etc.)
    katt_rp: torch.Tensor | None = None  # attitude P gain roll/pitch [1/s]
    katt_y: torch.Tensor | None = None   # attitude P gain yaw [1/s]

    @staticmethod
    def nominal(batch: int, device="cpu") -> "DynParams":
        t = lambda v: torch.full((batch,), float(v), device=device)
        return DynParams(
            twr=t(2.0), tau_w=t(0.05), tau_c=t(0.03), kd_lin=t(0.1),
            delay_steps=torch.full((batch,), 3, device=device, dtype=torch.long),
            thrust_gain=t(1.0),
        )

    @staticmethod
    def randomized(batch: int, device="cpu", g: torch.Generator | None = None) -> "DynParams":
        """Sample per-episode parameters uniformly in the default ranges."""
        u = lambda lo, hi: lo + (hi - lo) * torch.rand((batch,), device=device, generator=g)
        return DynParams(
            twr=u(2.0, 3.2),
            tau_w=u(0.015, 0.06),
            tau_c=u(0.010, 0.045),
            kd_lin=u(KD_LIN_FLOOR, 0.30),
            delay_steps=torch.randint(1, 5, (batch,), device=device, generator=g),
            thrust_gain=u(0.85, 1.15),
        )


@dataclass
class QuadState:
    p: torch.Tensor            # (B,3) m
    v: torch.Tensor            # (B,3) m/s
    q: torch.Tensor            # (B,4) body->world
    w: torch.Tensor            # (B,3) rad/s actual body rates
    thrust: torch.Tensor       # (B,)  produced mass-normalized thrust m/s^2
    cmd_fifo: torch.Tensor     # (B,D,4) pending delayed commands
    a_world: torch.Tensor = field(default=None)  # (B,3) last linear accel (for IMU)

    def detach(self) -> "QuadState":
        return QuadState(*(x.detach() if torch.is_tensor(x) else x for x in
                           (self.p, self.v, self.q, self.w, self.thrust,
                            self.cmd_fifo, self.a_world)))


class QuadCTBRDynamics:
    """Collective-thrust + body-rate plant.

    Each ``dt_ctrl`` control step integrates ``n_sub`` substeps of
    ``dt_sim = dt_ctrl / n_sub`` (semi-implicit Euler).
    """

    RATE_LIMIT = (4.0, 4.0, 2.0)  # rad/s clamp on commanded body rates (x,y,z)

    def __init__(self, dt_ctrl: float = 0.025, n_sub: int = 5, max_delay_steps: int = 5):
        self.dt_ctrl = dt_ctrl
        self.n_sub = n_sub
        self.dt_sim = dt_ctrl / n_sub
        self.max_delay = max_delay_steps

    def finalize_params(self, params: DynParams,
                        g: torch.Generator | None = None) -> DynParams:
        """Per-episode parameter hook. CTBR consumes nothing here; the
        attitude subclass draws its cascade gains."""
        return params

    # ------------------------------------------------------------- lifecycle
    def make_state(self, p, v, q, w, params: DynParams) -> QuadState:
        """Build an initial :class:`QuadState`. The command FIFO is prefilled
        with the hover thrust so ``t=0`` is not a free-fall artifact."""
        B = p.shape[0]
        dev = p.device
        hover = torch.full((B,), G, device=dev) / params.thrust_gain
        fifo = torch.zeros((B, self.max_delay, 4), device=dev)
        fifo[..., 0] = hover.unsqueeze(-1)
        return QuadState(p=p, v=v, q=quat_normalize(q), w=w,
                         thrust=hover * params.thrust_gain,
                         cmd_fifo=fifo, a_world=torch.zeros((B, 3), device=dev))

    # ----------------------------------------------------------------- step
    def step(self, s: QuadState, action: torch.Tensor, params: DynParams) -> QuadState:
        """Advance one control step. ``action`` (B,4) = ``[c, wx, wy, wz]``
        commanded NOW; the plant applies the FIFO-delayed command."""
        B = action.shape[0]
        dev = action.device
        rl = torch.tensor(self.RATE_LIMIT, device=dev)
        c_cmd = action[:, 0].clamp(0.0, 1.0e9)  # upper bound applied via twr below
        c_cmd = torch.minimum(c_cmd, params.twr * G)
        w_cmd = torch.max(torch.min(action[:, 1:], rl), -rl)
        cmd = torch.cat((c_cmd.unsqueeze(-1), w_cmd), dim=-1)

        # FIFO delay (per-sample depth): shift-in cmd, read at delay_steps-1
        fifo = torch.cat((cmd.unsqueeze(1), s.cmd_fifo[:, :-1]), dim=1)
        idx = (params.delay_steps - 1).clamp(0, self.max_delay - 1)
        applied = fifo[torch.arange(B, device=dev), idx]  # (B,4)
        c_app, w_app = applied[:, 0], applied[:, 1:]

        p, v, q, w, thrust = s.p, s.v, s.q, s.w, s.thrust
        a_world = s.a_world
        alpha_w = 1.0 - torch.exp(-self.dt_sim / params.tau_w)
        alpha_c = 1.0 - torch.exp(-self.dt_sim / params.tau_c)

        for _ in range(self.n_sub):
            # actuator lags: rates track command; thrust tracks command*gain
            w = w + alpha_w.unsqueeze(-1) * (w_app - w)
            thrust = thrust + alpha_c * (c_app * params.thrust_gain - thrust)
            # kinematics/dynamics
            q = quat_normalize(quat_mul(q, quat_exp_map(w, self.dt_sim)))
            thrust_world = quat_rotate(q, torch.stack(
                (torch.zeros_like(thrust), torch.zeros_like(thrust), -thrust), dim=-1))
            a_world = thrust_world + torch.tensor([0.0, 0.0, G], device=dev) \
                - params.kd_lin.unsqueeze(-1) * v
            v = v + a_world * self.dt_sim
            p = p + v * self.dt_sim

        return QuadState(p=p, v=v, q=q, w=w, thrust=thrust, cmd_fifo=fifo,
                         a_world=a_world)


class QuadAttitudeDynamics(QuadCTBRDynamics):
    """Attitude-setpoint + thrust plant.

    Action (B,4): ``[c, roll_sp, pitch_sp, yaw_sp]``
      - ``c``: mass-normalized collective thrust [m/s^2], clamped as CTBR.
      - ``roll_sp`` / ``pitch_sp``: tilt setpoints [rad], clamped
        ``+/-TILT_SP_LIMIT``.
      - ``yaw_sp``: absolute yaw setpoint [rad]; the cascade wraps the error
        (quaternion shortest path), so any value is legal.

    A PX4-style quaternion-P attitude cascade runs on top of the unchanged
    rate loop. Per sim substep::

        q_err = q^{-1} (x) q_sp                        (body-frame error)
        w_cmd = K_att * 2 * sign(q_err_w) * q_err_vec  (shortest path)
        w_cmd clamped at ATT_RATE_LIMIT

    then the existing rate-loop lag consumes ``w_cmd``. The FIFO transport
    delay applies to the setpoint; the cascade then runs on the current
    attitude. ``K_att = (katt_rp, katt_rp, katt_y)``.
    """

    # ------------------------------------------------------------- lifecycle
    def finalize_params(self, params: DynParams,
                        g: torch.Generator | None = None) -> DynParams:
        """Attach the attitude-cascade gains: nominal when ``g`` is ``None``,
        else drawn +/-KATT_DR per episode."""
        B = params.twr.shape[0]
        dev = params.twr.device
        if g is None:
            params.katt_rp = torch.full((B,), KATT_RP_NOM, device=dev)
            params.katt_y = torch.full((B,), KATT_Y_NOM, device=dev)
        else:
            u = lambda: (1.0 - KATT_DR + 2.0 * KATT_DR
                         * torch.rand((B,), generator=g)).to(dev)
            params.katt_rp = KATT_RP_NOM * u()
            params.katt_y = KATT_Y_NOM * u()
        return params

    def make_state(self, p, v, q, w, params: DynParams) -> QuadState:
        s = super().make_state(p, v, q, w, params)
        # Prefill the FIFO yaw slot with the current yaw: a zero yaw setpoint
        # is an absolute heading, so it must default to the spawn heading
        # (roll/pitch zeros are already the correct level setpoint).
        yaw0, _, _ = euler_zyx_from_quat(quat_normalize(q))
        s.cmd_fifo[..., 3] = yaw0.unsqueeze(-1)
        return s

    # ----------------------------------------------------------------- step
    def step(self, s: QuadState, action: torch.Tensor, params: DynParams) -> QuadState:
        """Advance one control step. ``action`` (B,4) = ``[c, roll_sp,
        pitch_sp, yaw_sp]`` commanded NOW; the plant applies the FIFO-delayed
        setpoint through the attitude cascade and the unchanged rate loop."""
        B = action.shape[0]
        dev = action.device
        arl = torch.tensor(ATT_RATE_LIMIT, device=dev)
        c_cmd = action[:, 0].clamp(0.0, 1.0e9)  # upper bound via twr below
        c_cmd = torch.minimum(c_cmd, params.twr * G)
        tilt_sp = action[:, 1:3].clamp(-TILT_SP_LIMIT, TILT_SP_LIMIT)
        cmd = torch.cat((c_cmd.unsqueeze(-1), tilt_sp, action[:, 3:4]), dim=-1)

        # FIFO delay (per-sample depth), exactly as the parent
        fifo = torch.cat((cmd.unsqueeze(1), s.cmd_fifo[:, :-1]), dim=1)
        idx = (params.delay_steps - 1).clamp(0, self.max_delay - 1)
        applied = fifo[torch.arange(B, device=dev), idx]  # (B,4)
        c_app = applied[:, 0]
        q_sp = quat_from_euler_zyx(applied[:, 3], applied[:, 2], applied[:, 1])
        katt_rp = params.katt_rp if params.katt_rp is not None \
            else torch.full_like(params.twr, KATT_RP_NOM)
        katt_y = params.katt_y if params.katt_y is not None \
            else torch.full_like(params.twr, KATT_Y_NOM)
        katt = torch.stack((katt_rp, katt_rp, katt_y), dim=-1)  # (B,3)

        p, v, q, w, thrust = s.p, s.v, s.q, s.w, s.thrust
        a_world = s.a_world
        alpha_w = 1.0 - torch.exp(-self.dt_sim / params.tau_w)
        alpha_c = 1.0 - torch.exp(-self.dt_sim / params.tau_c)

        for _ in range(self.n_sub):
            # attitude P loop at the substep rate: body-frame quaternion error
            # to the (delayed) setpoint, shortest path via the hemisphere sign;
            # output clamped like any rate command
            q_err = quat_mul(quat_conj(q), q_sp)
            sgn = torch.where(q_err[:, :1] < 0,
                              -torch.ones_like(q_err[:, :1]),
                              torch.ones_like(q_err[:, :1]))
            w_app = katt * (2.0 * sgn * q_err[:, 1:])
            w_app = torch.max(torch.min(w_app, arl), -arl)
            # rate loop + kinematics/dynamics: verbatim parent step body
            w = w + alpha_w.unsqueeze(-1) * (w_app - w)
            thrust = thrust + alpha_c * (c_app * params.thrust_gain - thrust)
            q = quat_normalize(quat_mul(q, quat_exp_map(w, self.dt_sim)))
            thrust_world = quat_rotate(q, torch.stack(
                (torch.zeros_like(thrust), torch.zeros_like(thrust), -thrust), dim=-1))
            a_world = thrust_world + torch.tensor([0.0, 0.0, G], device=dev) \
                - params.kd_lin.unsqueeze(-1) * v
            v = v + a_world * self.dt_sim
            p = p + v * self.dt_sim

        return QuadState(p=p, v=v, q=q, w=w, thrust=thrust, cmd_fifo=fifo,
                         a_world=a_world)
