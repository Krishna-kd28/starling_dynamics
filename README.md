# starling_dynamics

Batched quadrotor rigid-body dynamics and a strapdown IMU
model, in PyTorch. Self-contained (only `torch` + `numpy`), framework-agnostic.
Defaults are for a small quadrotor (ModalAI Starling 2).

The autopilot inner loop (rate controller + motors) is modeled as first-order
tracking of the commanded body rates / collective thrust, with a transport
delay on the command and linear rotor drag. Two command interfaces share the
same plant: collective-thrust + body-rate (CTBR), and attitude-setpoint +
thrust.

## Install

```bash
pip install torch numpy
# then put the package dir on PYTHONPATH, or:
pip install -e .
```

Everything runs on CPU. It is batched, so it also runs on GPU unchanged.

## Model

### State (batched over `B`)

| symbol | shape | meaning | frame / units |
|---|---|---|---|
| `p` | (B,3) | position | world, m |
| `v` | (B,3) | linear velocity | world, m/s |
| `q` | (B,4) | attitude quaternion, body→world, `(w,x,y,z)` | unit |
| `w` | (B,3) | body angular rates | body, rad/s |
| `thrust` | (B,) | produced mass-normalized collective thrust | m/s² |
| `cmd_fifo` | (B,D,4) | queue of pending (delayed) commands | — |
| `a_world` | (B,3) | last linear acceleration (used by the IMU) | world, m/s² |

### Frames & conventions

- **World**: metres, `z` points **DOWN** (NED-like). Gravity `g = [0, 0, +G]`,
  `G = 9.81`.
- **Body**: FRD (x forward, y right, z down). Collective thrust acts along
  `-z_body`.
- **Quaternion**: `(w, x, y, z)`, scalar-first, Hamilton product, maps
  body→world. Euler helpers use intrinsic **Z-Y-X** (yaw, pitch, roll),
  matching `scipy Rotation.from_euler('ZYX', ...)`.

### Continuous dynamics

```
ṗ = v
v̇ = R(q)·[0, 0, -T]  +  [0, 0, G]  -  k_d · v      (thrust, gravity, linear drag)
q̇ = ½ · q ⊗ [0, ω]
Ṫ  = (T_cmd - T) / τ_c                              (thrust first-order lag)
ω̇  = (ω_cmd - ω) / τ_ω                              (rate-loop first-order lag)
```

`T` is the produced mass-normalized thrust (`T_cmd = gain · c`); `k_d` is a
linear rotor-drag coefficient; `ω_cmd` is the (delayed) commanded body rate.

### Discretization

Each control step `dt_ctrl` integrates `n_sub` substeps of
`dt_sim = dt_ctrl / n_sub` with **semi-implicit Euler** (update `v` then `p`).
The first-order lags use the exact discrete gain
`α = 1 - exp(-dt_sim / τ)`; the quaternion is advanced by the exponential map
`q ← normalize(q ⊗ exp(½ ω dt_sim))`. Defaults: `dt_ctrl = 0.025 s` (40 Hz),
`n_sub = 5` → `dt_sim = 0.005 s` (200 Hz).

### Transport delay

The command is pushed into a FIFO and the plant applies the entry from
`delay_steps` control steps ago (a per-sample integer dead time). The FIFO is
prefilled with the hover command so `t = 0` is not a free-fall artifact.

### Action spaces

| mode | class | action `(B,4)` | use when |
|---|---|---|---|
| CTBR | `QuadCTBRDynamics` | `[c, ωx_cmd, ωy_cmd, ωz_cmd]` | you command body **rates** directly |
| Attitude | `QuadAttitudeDynamics` | `[c, roll_sp, pitch_sp, yaw_sp]` | you command an **attitude** setpoint and let an onboard attitude controller produce the rates |

- `c`: mass-normalized collective thrust [m/s²], clamped to `[0, twr·G]`.
  (Hover ⇒ `c ≈ G`.)
- CTBR rates [rad/s] are clamped to `QuadCTBRDynamics.RATE_LIMIT = (4, 4, 2)`.
- Attitude `roll_sp/pitch_sp` [rad] are clamped to `TILT_SP_LIMIT = 0.35`;
  `yaw_sp` [rad] is **absolute** (the cascade takes the shortest-path error).
- Attitude mode adds a PX4-style quaternion-P cascade on top of the same rate
  loop: `ω_cmd = K_att · 2·sign(q_err_w)·q_err_vec`, re-evaluated each substep
  against the current attitude and clamped to
  `ATT_RATE_LIMIT = (2.269, 2.269, 2.618)` rad/s. CTBR mode is unaffected.

## IMU

`ImuSim.read(state)` returns `{"gyro", "accel", "tilt"}`:

```
gyro  = ω_body + bias_g + noise                              [rad/s]
accel = specific force = Rᵀ(q) · (a_world - g) + bias_a + noise   [m/s²]
tilt  = (roll, pitch) + slow random-walk drift + noise        [rad]   (optional)
```

`accel` is **specific force** in body axes: at a level hover it reads
`[0, 0, -G]` (the upward thrust reaction, since up = `-z` in a z-down frame) and
in free fall it reads `≈ 0`. Because thrust is along `-z_body`, the accel x/y
channels carry the aerodynamic drag. Biases are per-episode constants (redrawn
on `reset`); noise is white per read. `tilt` models an IMU-only roll/pitch
estimate (yaw is not observable without external aiding). Default read rate =
control rate (40 Hz).

## Default parameters (Starling 2)

Physical constants (public sources: ModalAI Starling 2 datasheet; PX4
`D0014_Starling_2` params; TDK ICM-42688-P datasheet):

| quantity | value |
|---|---|
| take-off mass | 0.285 kg |
| motor-to-motor diagonal | 211 mm |
| propellers / motors / battery | 120 mm / 1504 3000 KV / 2S |
| moment of inertia (roll, approx) | ≈ 4e-4 kg·m² |
| hover throttle fraction (`MPC_THR_HOVER`) | 0.34 |
| thrust-model factor (`THR_MDL_FAC`) | 0.9 |
| max thrust-to-weight | ≈ 2.6–2.9 |
| ESC RPM range | 2000–15000 |
| attitude P gains `MC_ROLL_P / MC_PITCH_P / MC_YAW_P` | 16.0 / 16.0 / 2.8 (1/s) |
| attitude rate limits `MC_*RATE_MAX` | 130 / 130 / 150 °/s |
| flight-controller IMU | TDK InvenSense ICM-42688-P |

`DynParams` — `nominal(B)` values and `randomized(B, g)` per-episode ranges:

| field | nominal | randomized range | meaning |
|---|---|---|---|
| `twr` | 2.0 | [2.0, 3.2] | thrust-to-weight (max thrust `= twr·G`) |
| `tau_w` | 0.05 s | [0.015, 0.060] | rate-loop time constant |
| `tau_c` | 0.03 s | [0.010, 0.045] | thrust (motor) time constant |
| `kd_lin` | 0.1 | [0.03, 0.30] s⁻¹ | linear rotor-drag coefficient |
| `delay_steps` | 3 | {1,2,3,4} | transport delay in control steps (×25 ms) |
| `thrust_gain` | 1.0 | [0.85, 1.15] | multiplicative thrust-map error |
| `katt_rp` | 16.0 | ±20% | attitude-P roll/pitch gain (attitude mode) |
| `katt_y` | 2.8 | ±20% | attitude-P yaw gain (attitude mode) |

`ImuParams` defaults (per-read white noise; per-episode constant bias):
`gyro_noise = 0.005` rad/s, `accel_noise = 0.10` m/s²,
`gyro_bias σ = 0.02` rad/s, `accel_bias σ = 0.2` m/s²,
`tilt_noise = 0.005` rad, `tilt_drift_rate = 0.002` rad/√s.

**Deployment note.** To command a PX4 vehicle, convert the mass-normalized
thrust to normalized throttle with `thrust01 = MPC_THR_HOVER · c / G`
(hover `c = G` → `0.34`).

## Usage

Plain forward simulation:

```python
import torch
from starling_dynamics import DynParams, G, QuadCTBRDynamics, quat_from_euler_zyx

B = 4
dyn = QuadCTBRDynamics(dt_ctrl=0.025, n_sub=5)
params = DynParams.nominal(B)                       # or .randomized(B, g=torch.Generator().manual_seed(0))
q = quat_from_euler_zyx(torch.zeros(B), torch.zeros(B), torch.zeros(B))
state = dyn.make_state(torch.zeros(B, 3), torch.zeros(B, 3), q, torch.zeros(B, 3), params)

hover = torch.tensor([G, 0.0, 0.0, 0.0]).expand(B, 4)
for _ in range(100):
    state = dyn.step(state, hover, params)          # [c, wx, wy, wz]
```

Attitude mode — same API, different action:

```python
from starling_dynamics import QuadAttitudeDynamics
adyn = QuadAttitudeDynamics()
params = adyn.finalize_params(DynParams.nominal(B))            # attaches attitude gains
action = torch.tensor([G, 0.1, 0.0, 0.5]).expand(B, 4)        # [c, roll_sp, pitch_sp, yaw_sp]
state = adyn.step(state, action, params)
```

IMU:

```python
from starling_dynamics import ImuParams, ImuSim
imu = ImuSim(ImuParams.nominal(B), dyn.dt_ctrl)               # or ImuParams.randomized(B, g=...)
reading = imu.read(state)                                     # {"gyro","accel","tilt"}
```

## API reference

- **Dynamics**: `QuadCTBRDynamics(dt_ctrl, n_sub, max_delay_steps)`,
  `QuadAttitudeDynamics(...)`. Methods: `make_state(p, v, q, w, params) → QuadState`,
  `step(state, action, params) → QuadState`, `finalize_params(params, g=None)`.
- **State / params**: `QuadState` (dataclass),
  `DynParams.nominal(B)`, `DynParams.randomized(B, g)`.
- **IMU**: `ImuParams.nominal(B)`, `ImuParams.randomized(B, g)`,
  `ImuSim(params, dt)` with `.read(state, g=None)` and `.reset(params=None)`.
- **Quaternions**: `quat_mul`, `quat_conj`, `quat_rotate`, `quat_rotate_inv`,
  `quat_normalize`, `quat_exp_map`, `quat_from_euler_zyx`, `euler_zyx_from_quat`.
- **Constants**: `G`, `TILT_SP_LIMIT`, `KATT_RP_NOM`, `KATT_Y_NOM`,
  `ATT_RATE_LIMIT`, `KATT_DR`, `KD_LIN_FLOOR`; `QuadCTBRDynamics.RATE_LIMIT`.

## Modeling assumptions & fidelity

This is a control-oriented plant, not a full aeromechanical simulation. What it
models, and what it deliberately abstracts:

- **Exact / standard.** Rigid-body translation and quaternion kinematics;
  specific-force accelerometer model `Rᵀ(a−g)` (sign, frame, gravity, units
  verified); exact first-order-lag discretization; per-sample transport delay;
  PX4-style quaternion-P attitude cascade (shortest-path, correct hemisphere and
  clamp placement).
- **Approximate.** The rate/thrust loops are lumped first-order lags (no
  overshoot, no per-axis asymmetry, no integral disturbance rejection). Drag is
  an isotropic world-frame `−k_d·v` (real rotor drag is body-frame and mildly
  anisotropic). Thrust→acceleration is treated as linear via `thrust_gain`.
- **Omitted.** Rigid-body rotational dynamics (`J`, `ω×Jω`, gyroscopic terms)
  are hidden behind the rate-loop abstraction; motor RPM saturation /
  control-allocation coupling; ground/wall aerodynamic effects; accelerometer
  vibration (the dominant real accel disturbance — the white `accel_noise` is
  not a substitute); in-run IMU bias drift (bias is constant per episode).
- **For sim2real.** The default parameters are reasonable Starling-2 values but
  are not system-identified; measure `tau_w`, `tau_c`, `thrust_gain`,
  hover throttle, drag, latency, and the attitude gains on the real vehicle and
  re-center the ranges before trusting the model for transfer. If your policy
  consumes the accelerometer, bench-measure prop-vibration and widen
  `accel_noise` accordingly.
