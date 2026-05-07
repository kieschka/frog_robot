import math
import os
import csv
from controller import Robot

CONTROLLER = "pid_ff"


PARAMS = {
    "time_step_ms"        : 4,

    "depth_target_m"      : 0.80,
    "pitch_target_rad"    : -0.175,
    "pool_half_x_m"       : 4.0,
    "pool_half_y_m"       : 4.0,
    "wall_safe_m"         : 0.5,

    "waypoints_xy"        : [
        ( 2.5,  2.5),
        ( 2.5, -2.5),
        (-2.5, -2.5),
        (-2.5,  2.5),
    ],
    "waypoint_radius_m"   : 0.6,
    "duration_power_s"    : 0.30,
    "duration_glide_s"    : 0.40,
    "duration_recovery_s" : 0.30,

    "pose_extended" : {
        "hip"   : math.radians( 10.0),
        "knee"  : math.radians(-30.0),
        "ankle" : math.radians(-30.0),
    },
    "pose_tucked" : {
        "hip"   : math.radians( 55.0),
        "knee"  : math.radians(-75.0),
        "ankle" : math.radians(-25.0),
    },

    "knee_phase_lead_s" : 0.030,
    "ankle_phase_lag_s" : 0.030,
    "pid_yaw"   : {"kp": 0.40, "ki": 0.02, "kd": 0.10,
                   "u_min": -0.60, "u_max": 0.60,
                   "i_max": 0.20,  "deriv_tau": 0.20},
    "pid_depth" : {"kp": 0.30, "ki": 0.02, "kd": 0.08,
                   "u_min": -0.30, "u_max": 0.30,
                   "i_max": 0.15,  "deriv_tau": 0.20},
    "pid_pitch" : {"kp": 0.40, "ki": 0.02, "kd": 0.12,
                   "u_min": -0.50, "u_max": 0.50,
                   "i_max": 0.20,  "deriv_tau": 0.20},
    "pid_roll"  : {"kp": 0.20, "ki": 0.00, "kd": 0.08,
                   "u_min": -0.20, "u_max": 0.20,
                   "i_max": 0.10,  "deriv_tau": 0.25},

    "smc_yaw"   : {"lambda": 6.0, "k": 0.50, "phi": 0.05,
                   "u_min": -0.60, "u_max": 0.60},
    "smc_depth" : {"lambda": 5.0, "k": 0.20, "phi": 0.05,
                   "u_min": -0.30, "u_max": 0.30},
    "smc_pitch" : {"lambda": 6.0, "k": 0.40, "phi": 0.05,
                   "u_min": -0.50, "u_max": 0.50},
    "smc_roll"  : {"lambda": 6.0, "k": 0.15, "phi": 0.05,
                   "u_min": -0.20, "u_max": 0.20},

    "joint_pd" : {
        "hip"   : {"kp": 6.0,  "kd": 0.40},
        "knee"  : {"kp": 4.0,  "kd": 0.30},
        "ankle" : {"kp": 1.0,  "kd": 0.08},
    },

    "torque_alloc" : {
        "yaw_to_hip"      : 1.00,
        "pitch_to_hip"    : 0.30,
        "pitch_to_knee"   : 0.70,
        "depth_to_ankle"  : 1.00,
        "roll_to_ankle"   : 1.00,
    },

    "warmup_hold_s"  : 0.5,
    "warmup_total_s" : 1.5,
}


def clamp(v, lo, hi): return max(lo, min(hi, v))

def smoothstep(s):
    s = clamp(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)

def quintic(s):
    s = clamp(s, 0.0, 1.0)
    return 10*s**3 - 15*s**4 + 6*s**5

def wrap_to_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def lerp(a, b, s): return a + (b - a) * s

def lerp_pose(p1, p2, s):
    return {k: lerp(p1[k], p2[k], s) for k in p1}

class PIDController:
    def __init__(self, kp, ki, kd, dt_s, u_min, u_max, i_max, deriv_tau):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt_s
        self.u_min, self.u_max = u_min, u_max
        self.i_max = i_max
        self.alpha = dt_s / max(deriv_tau, 1e-6)
        self.integ = 0.0; self.prev_e = 0.0; self.deriv = 0.0

    def step(self, e):
        self.integ = clamp(self.integ + e*self.dt, -self.i_max, self.i_max)
        d_raw = (e - self.prev_e) / self.dt
        self.deriv += self.alpha * (d_raw - self.deriv)
        self.prev_e = e
        u = self.kp*e + self.ki*self.integ + self.kd*self.deriv
        u_sat = clamp(u, self.u_min, self.u_max)
        if self.ki != 0:
            self.integ += (u_sat - u) / self.ki * 0.5
        return u_sat


class PIDWithFeedforward(PIDController):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ff = 0.0
    def set_feedforward(self, ff):
        self.ff = ff
    def step(self, e):
        u_pid = super().step(e)
        return clamp(u_pid + self.ff, self.u_min, self.u_max)


class SlidingModeController:
    def __init__(self, lam, k, phi, dt_s, u_min, u_max):
        self.lam = lam
        self.k = k
        self.phi = max(phi, 1e-6)
        self.dt = dt_s
        self.u_min, self.u_max = u_min, u_max
        self.prev_e = 0.0
    def step(self, e):
        de = (e - self.prev_e) / self.dt
        self.prev_e = e
        s = de + self.lam * e
        if abs(s) > self.phi:
            sat_s = math.copysign(1.0, s)
        else:
            sat_s = s / self.phi
        u = -self.k * sat_s
        return clamp(u, self.u_min, self.u_max)


def build_controller(kind, channel, dt_s):
    p = PARAMS
    if kind in ("pid", "pid_ff"):
        cfg = p["pid_" + channel]
        cls = PIDController if kind == "pid" else PIDWithFeedforward
        return cls(cfg["kp"], cfg["ki"], cfg["kd"], dt_s,
                   cfg["u_min"], cfg["u_max"], cfg["i_max"], cfg["deriv_tau"])
    elif kind == "smc":
        cfg = p["smc_" + channel]
        return SlidingModeController(cfg["lambda"], cfg["k"], cfg["phi"], dt_s,
                                     cfg["u_min"], cfg["u_max"])
    raise ValueError(kind)

def gait_state(t_cycle, p, side='R', amp_scale=1.0):
    T_p = p["duration_power_s"]
    T_g = p["duration_glide_s"]
    T_r = p["duration_recovery_s"]
    T_total = T_p + T_g + T_r

    if side == 'L':
        t_cycle = t_cycle + T_total * 0.5

    t = t_cycle % T_total

    pose_t = p["pose_tucked"]
    pose_e = p["pose_extended"]
    knee_lead = p["knee_phase_lead_s"]
    ankle_lag = p["ankle_phase_lag_s"]

    amp_scale = max(0.3, min(1.0, amp_scale))
    pose_e_scaled = {
        "hip"   : lerp(pose_t["hip"],   pose_e["hip"],   amp_scale),
        "knee"  : lerp(pose_t["knee"],  pose_e["knee"],  amp_scale),
        "ankle" : lerp(pose_t["ankle"], pose_e["ankle"], amp_scale),
    }

    if t < T_p:
        s_hip = quintic(t / T_p)
        t_knee = t + knee_lead
        s_knee = quintic(t_knee / T_p)
        t_ankle = max(0.0, t - ankle_lag)
        s_ankle = quintic(t_ankle / T_p)
        joint_pose = {
            "hip"   : lerp(pose_t["hip"],   pose_e_scaled["hip"],   s_hip),
            "knee"  : lerp(pose_t["knee"],  pose_e_scaled["knee"],  s_knee),
            "ankle" : lerp(pose_t["ankle"], pose_e_scaled["ankle"], s_ankle),
        }
        return "power", joint_pose

    elif t < T_p + T_r:
        tau = t - T_p
        s_hip = quintic(tau / T_r)
        t_knee = tau + knee_lead
        s_knee = quintic(t_knee / T_r)
        t_ankle = max(0.0, tau - ankle_lag)
        s_ankle = quintic(t_ankle / T_r)
        s_a_norm = max(0.0, min(1.0, t_ankle / T_r))
        env = quintic(2.0 * s_a_norm) if s_a_norm < 0.5 else quintic(2.0 - 2.0 * s_a_norm)
        feather = math.radians(25.0) * env
        joint_pose = {
            "hip"   : lerp(pose_e_scaled["hip"],   pose_t["hip"],   s_hip),
            "knee"  : lerp(pose_e_scaled["knee"],  pose_t["knee"],  s_knee),
            "ankle" : lerp(pose_e_scaled["ankle"], pose_t["ankle"], s_ankle) + feather,
        }
        return "recovery", joint_pose

    else:
        tau_g = t - T_p - T_r
        s_g = max(0.0, min(1.0, tau_g / T_g))
        env_g = quintic(2.0 * s_g) if s_g < 0.5 else quintic(2.0 - 2.0 * s_g)
        glide_feather = math.radians(30.0) * env_g
        joint_pose = dict(pose_t)
        joint_pose["ankle"] = pose_t["ankle"] + glide_feather
        return "glide", joint_pose


class Joint:
    def __init__(self, robot, name, dt_ms, kp_joint, kd_joint):
        self.name = name
        self.motor  = robot.getDevice(name)
        self.sensor = robot.getDevice(name + "_sensor")
        if self.sensor is not None:
            self.sensor.enable(dt_ms)
        try:
            self.motor.setPosition(float('inf'))
            self.motor.setVelocity(0.0)
            self.motor.setTorque(0.0)
        except Exception:
            pass
        try:
            self.q_min = self.motor.getMinPosition()
            self.q_max = self.motor.getMaxPosition()
        except Exception:
            self.q_min, self.q_max = -3.14, 3.14
        try:
            self.tau_max = self.motor.getMaxTorque()
        except Exception:
            self.tau_max = 4.0
        try:
            self.v_max = self.motor.getMaxVelocity()
        except Exception:
            self.v_max = 6.0

        self.kp = kp_joint
        self.kd = kd_joint
        self.dt = dt_ms * 1e-3
        self.q_dot_filt = 0.0
        self.deriv_alpha = self.dt / 0.030  # τ = 30 мс
        self.q_prev = 0.0
        self.q_ref_filt = 0.0
        self.q_ref_dot_filt = 0.0
        self.first = True
        self.tau_last = 0.0

    def get_position(self):
        if self.sensor is None:
            return 0.0
        try:
            return self.sensor.getValue()
        except Exception:
            return 0.0

    def step(self, q_ref, tau_correct=0.0):
        q = self.get_position()
        if self.first:
            self.q_prev = q
            self.q_ref_filt = q
            self.q_dot_filt = 0.0
            self.first = False
        max_dq = self.v_max * self.dt
        q_ref_step = clamp(q_ref - self.q_ref_filt, -max_dq, max_dq)
        q_ref_new = self.q_ref_filt + q_ref_step
        q_ref_new = clamp(q_ref_new, self.q_min + 1e-3, self.q_max - 1e-3)
        q_ref_dot = (q_ref_new - self.q_ref_filt) / self.dt
        self.q_ref_filt = q_ref_new
        q_dot_raw = (q - self.q_prev) / self.dt
        self.q_dot_filt += self.deriv_alpha * (q_dot_raw - self.q_dot_filt)
        self.q_prev = q
        tau_track = self.kp * (self.q_ref_filt - q) + self.kd * (q_ref_dot - self.q_dot_filt)
        tau = clamp(tau_track + tau_correct, -self.tau_max, self.tau_max)

        try:
            self.motor.setTorque(tau)
        except Exception:
            pass

        self.tau_last = tau
        return tau

def main():
    p = PARAMS
    dt_ms = p["time_step_ms"]
    dt_s  = dt_ms * 1e-3

    robot = Robot()
    gps   = robot.getDevice("gps");   gps.enable(dt_ms)
    imu   = robot.getDevice("imu");   imu.enable(dt_ms)
    gyro  = robot.getDevice("gyro");  gyro.enable(dt_ms)
    accel = robot.getDevice("accel"); accel.enable(dt_ms)

    JOINT_INFO = [
        ("hipR",   "hip"),   ("kneeR",  "knee"),  ("ankleR", "ankle"),
        ("hipL",   "hip"),   ("kneeL",  "knee"),  ("ankleL", "ankle"),
    ]
    joints = {}
    for name, kind in JOINT_INFO:
        gains = p["joint_pd"][kind]
        joints[name] = Joint(robot, name, dt_ms, gains["kp"], gains["kd"])

    SETTLING_STEPS = 5
    for _ in range(SETTLING_STEPS):
        if robot.step(dt_ms) == -1:
            return

    q_ref = {}
    for jname in joints:
        q_actual = joints[jname].get_position()
        q_ref[jname] = q_actual
        joints[jname].q_prev = q_actual
        joints[jname].q_ref_filt = q_actual
        joints[jname].first = False

    pose_start_R = {"hip":   joints["hipR"].get_position(),
                    "knee":  joints["kneeR"].get_position(),
                    "ankle": joints["ankleR"].get_position()}
    pose_start_L = {"hip":   joints["hipL"].get_position(),
                    "knee":  joints["kneeL"].get_position(),
                    "ankle": joints["ankleL"].get_position()}

    yaw_ctrl   = build_controller(CONTROLLER, "yaw",   dt_s)
    depth_ctrl = build_controller(CONTROLLER, "depth", dt_s)
    pitch_ctrl = build_controller(CONTROLLER, "pitch", dt_s)
    roll_ctrl  = build_controller(CONTROLLER, "roll",  dt_s)

    alloc = p["torque_alloc"]

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, "frog_log_" + CONTROLLER + ".csv")
    with open(log_path, "w", newline="") as log_f:
        log_w = csv.writer(log_f)
        log_w.writerow([
            "t_s", "controller", "phase",
            "x", "y", "z", "roll", "pitch", "yaw",
            "wx", "wy", "wz",
            "des_yaw", "yaw_err", "u_yaw_Nm",
            "depth_err", "u_depth_Nm",
            "pitch_err", "u_pitch_Nm",
            "roll_err", "u_roll_Nm",
            "hipR_ref", "kneeR_ref", "ankleR_ref",
            "hipL_ref", "kneeL_ref", "ankleL_ref",
            "tau_hipR", "tau_kneeR", "tau_ankleR",
            "tau_hipL", "tau_kneeL", "tau_ankleL",
            "wp_idx", "wp_x", "wp_y", "dist_to_wp",
        ])

        T_cycle = (p["duration_power_s"] + p["duration_glide_s"] + p["duration_recovery_s"])
        print("=" * 60)
        print("frog_controller V14 — TORQUE-CONTROL CASCADE")
        print("=" * 60)
        print(f"Controller (outer loop): {CONTROLLER}")
        print(f"Inner loop: per-joint PD on top of setTorque")
        print(f"Cycle: {T_cycle:.2f} s, freq {1.0/T_cycle:.2f} Hz")
        print(f"Target depth: {p['depth_target_m']:.2f} m, "
              f"target pitch: {math.degrees(p['pitch_target_rad']):.0f}°")
        print(f"Motor max torques:  hip={joints['hipR'].tau_max:.2f},  "
              f"knee={joints['kneeR'].tau_max:.2f},  ankle={joints['ankleR'].tau_max:.2f}  N·m")
        print("=" * 60)

        t = 0.0
        wp_idx = 0
        waypoints = list(p["waypoints_xy"])
        n_wp = len(waypoints)
        warmup_hold = p["warmup_hold_s"]
        warmup_s    = p["warmup_total_s"]
        soft_start_s        = 1.5
        warmup_hold_end     = soft_start_s + warmup_hold
        warmup_ramp_end     = warmup_hold_end + (warmup_s - warmup_hold)
        outer_loop_ramp_end = warmup_ramp_end + 1.5

        while robot.step(dt_ms) != -1:
            t += dt_s

            x, y, z = gps.getValues()
            roll, pitch, yaw = imu.getRollPitchYaw()
            wx, wy, wz = gyro.getValues()

            sensor_ok = (
                all(math.isfinite(v) for v in (x, y, z, roll, pitch, yaw, wx, wy, wz))
                and abs(x) < 50 and abs(y) < 50 and -1.0 <= z <= 5.0
            )
            if not sensor_ok:
                continue

            wp_x, wp_y = waypoints[wp_idx]
            dx, dy = wp_x - x, wp_y - y
            d_to_wp = math.hypot(dx, dy)
            if d_to_wp < p["waypoint_radius_m"]:
                wp_idx = (wp_idx + 1) % n_wp
                wp_x, wp_y = waypoints[wp_idx]

            cl = min(p["pool_half_x_m"] - x, x + p["pool_half_x_m"],
                     p["pool_half_y_m"] - y, y + p["pool_half_y_m"])
            tgt_x, tgt_y = (0.0, 0.0) if cl < p["wall_safe_m"] else (wp_x, wp_y)

            des_yaw   = math.atan2(tgt_y - y, tgt_x - x)
            yaw_err   = wrap_to_pi(des_yaw - yaw)
            depth_err = z - p["depth_target_m"]
            pitch_err = pitch - p["pitch_target_rad"]
            roll_err  = -roll

            if CONTROLLER == "pid_ff":
                yaw_ctrl.set_feedforward(0.0)
                depth_ctrl.set_feedforward(0.0)
                pitch_ctrl.set_feedforward(0.0)
                roll_ctrl.set_feedforward(0.0)

            if t < warmup_ramp_end:
                for ctrl in (yaw_ctrl, depth_ctrl, pitch_ctrl, roll_ctrl):
                    if hasattr(ctrl, "integ"):  ctrl.integ = 0.0
                    if hasattr(ctrl, "deriv"):  ctrl.deriv = 0.0
                    ctrl.prev_e = 0.0
                u_yaw = u_depth = u_pitch = u_roll = 0.0
            else:
                u_yaw   = yaw_ctrl.step(yaw_err)
                u_depth = depth_ctrl.step(depth_err)
                u_pitch = pitch_ctrl.step(pitch_err)
                u_roll  = roll_ctrl.step(roll_err)

            if t < soft_start_s:
                ramp = quintic(t / soft_start_s)
                joint_pose_R = lerp_pose(pose_start_R, p["pose_extended"], ramp)
                joint_pose_L = lerp_pose(pose_start_L, p["pose_extended"], ramp)
                phase = "soft_start"
            elif t < warmup_hold_end:
                joint_pose_R = dict(p["pose_extended"])
                joint_pose_L = dict(p["pose_extended"])
                phase = "warmup_hold"
            elif t < warmup_ramp_end:
                dur = max(warmup_ramp_end - warmup_hold_end, 1e-3)
                ramp = quintic((t - warmup_hold_end) / dur)
                _, pose_R_target = gait_state(0.0, p, 'R')
                _, pose_L_target = gait_state(0.0, p, 'L')
                joint_pose_R = lerp_pose(p["pose_extended"], pose_R_target, ramp)
                joint_pose_L = lerp_pose(p["pose_extended"], pose_L_target, ramp)
                phase = "warmup_ramp"
            else:
                t_cycle = t - warmup_ramp_end
                phase_R, joint_pose_R = gait_state(t_cycle, p, 'R')
                phase_L, joint_pose_L = gait_state(t_cycle, p, 'L')
                phase = phase_R
            if t < warmup_ramp_end:
                outer_gain = 0.0
            elif t < outer_loop_ramp_end:
                outer_gain = quintic((t - warmup_ramp_end) / 1.5)
            else:
                outer_gain = 1.0
            tau_correct = {
                "hipR":   outer_gain * (+u_yaw   * alloc["yaw_to_hip"]
                                        + u_pitch * alloc["pitch_to_hip"]),
                "hipL":   outer_gain * (-u_yaw   * alloc["yaw_to_hip"]
                                        + u_pitch * alloc["pitch_to_hip"]),
                "kneeR":  outer_gain * u_pitch  * alloc["pitch_to_knee"],
                "kneeL":  outer_gain * u_pitch  * alloc["pitch_to_knee"],
                "ankleR": outer_gain * (u_depth  * alloc["depth_to_ankle"]
                                        + u_roll * alloc["roll_to_ankle"]),
                "ankleL": outer_gain * (u_depth  * alloc["depth_to_ankle"]
                                        - u_roll * alloc["roll_to_ankle"]),
            }
            q_ref["hipR"]   = joint_pose_R["hip"]
            q_ref["kneeR"]  = joint_pose_R["knee"]
            q_ref["ankleR"] = joint_pose_R["ankle"]
            q_ref["hipL"]   = joint_pose_L["hip"]
            q_ref["kneeL"]  = joint_pose_L["knee"]
            q_ref["ankleL"] = joint_pose_L["ankle"]

            tau_applied = {}
            for jname in ("hipR", "kneeR", "ankleR", "hipL", "kneeL", "ankleL"):
                tau_applied[jname] = joints[jname].step(q_ref[jname], tau_correct[jname])
            log_w.writerow([
                "%.3f" % t, CONTROLLER, phase,
                x, y, z, roll, pitch, yaw,
                wx, wy, wz,
                des_yaw, yaw_err, u_yaw,
                depth_err, u_depth,
                pitch_err, u_pitch,
                roll_err, u_roll,
                q_ref["hipR"], q_ref["kneeR"], q_ref["ankleR"],
                q_ref["hipL"], q_ref["kneeL"], q_ref["ankleL"],
                tau_applied["hipR"], tau_applied["kneeR"], tau_applied["ankleR"],
                tau_applied["hipL"], tau_applied["kneeL"], tau_applied["ankleL"],
                wp_idx, wp_x, wp_y, d_to_wp,
            ])


if __name__ == "__main__":
    main()
