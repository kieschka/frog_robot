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

    "duration_power_s"      : 0.10,   
    "duration_glide_s"      : 0.50,    
    "duration_recovery_s"   : 0.40,    

    "pose_extended" : {
        "hip"          : math.radians( 10.0),
        "knee"         : math.radians(-30.0),
        "ankle"        : math.radians(-30.0),
    },
    "pose_tucked" : {
        "hip"          : math.radians( 55.0),
        "knee"         : math.radians(-75.0),
        "ankle"        : math.radians(-25.0),
    },

    "knee_phase_lead_s" : 0.030,
    "ankle_phase_lag_s" : 0.030,


    "pid_yaw"   : {"kp": 0.60, "ki": 0.02, "kd": 0.15,
                   "u_min": -0.50, "u_max": 0.50,
                   "i_max": 0.20,  "deriv_tau": 0.20},
    "pid_depth" : {"kp": 0.20, "ki": 0.01, "kd": 0.10,
                   "u_min": -0.20, "u_max": 0.20,
                   "i_max": 0.10,  "deriv_tau": 0.20},
    "pid_pitch" : {"kp": 0.40, "ki": 0.02, "kd": 0.15,
                   "u_min": -0.30, "u_max": 0.30,
                   "i_max": 0.15,  "deriv_tau": 0.20},
    "pid_roll"  : {"kp": 0.20, "ki": 0.00, "kd": 0.08,
                   "u_min": -0.15, "u_max": 0.15,
                   "i_max": 0.08,  "deriv_tau": 0.25},

    "smc_yaw"   : {"lambda": 12.0, "k": 0.30, "phi": 0.02,
                   "u_min": -0.30, "u_max": 0.30},
    "smc_depth" : {"lambda": 8.0, "k": 0.20, "phi": 0.02,
                   "u_min": -0.20, "u_max": 0.20},
    "smc_pitch" : {"lambda": 12.0, "k": 0.30, "phi": 0.02,
                   "u_min": -0.30, "u_max": 0.30},
    "smc_roll"  : {"lambda": 12.0, "k": 0.15, "phi": 0.02,
                   "u_min": -0.15, "u_max": 0.15},

    "hip_bias_yaw_gain"   : 0.30,
    "diff_power_gain"     : 0.5,
    "differential_power_gain" : 0.50,   
    "knee_bias_pitch_gain": 0.25,
    "hip_bias_pitch_gain" : 0.15,
    "ankle_bias_depth_gain": 0.30,

    "warmup_hold_s"     : 0.5,    
    "warmup_total_s"    : 1.5,    
}

def clamp(v, lo, hi): return max(lo, min(hi, v))

def smoothstep(s):
    """Smooth interpolation 0 to 1 with zero velocity at endpoints."""
    s = clamp(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)

def quintic(s):
    """Quintic blend - zero velocity AND acceleration at endpoints (Richards 2010)."""
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
        "hip":   lerp(pose_t["hip"],   pose_e["hip"],   amp_scale),
        "knee":  lerp(pose_t["knee"],  pose_e["knee"],  amp_scale),
        "ankle": lerp(pose_t["ankle"], pose_e["ankle"], amp_scale),
    }

    if t < T_p:
        s_hip = quintic(t / T_p)
        t_knee = t + knee_lead
        s_knee = quintic(t_knee / T_p)
        t_ankle = max(0.0, t - ankle_lag)
        s_ankle = quintic(t_ankle / T_p)
        joint_pose = {
            "hip":   lerp(pose_t["hip"],   pose_e_scaled["hip"],   s_hip),
            "knee":  lerp(pose_t["knee"],  pose_e_scaled["knee"],  s_knee),
            "ankle": lerp(pose_t["ankle"], pose_e_scaled["ankle"], s_ankle),
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
        if s_a_norm < 0.5:
            env = quintic(2.0 * s_a_norm)
        else:
            env = quintic(2.0 - 2.0 * s_a_norm)
        feather = math.radians(70.0) * env

        joint_pose = {
            "hip":   lerp(pose_e_scaled["hip"],   pose_t["hip"],   s_hip),
            "knee":  lerp(pose_e_scaled["knee"],  pose_t["knee"],  s_knee),
            "ankle": lerp(pose_e_scaled["ankle"], pose_t["ankle"], s_ankle) + feather,
        }
        return "recovery", joint_pose

    else:
        tau_g = t - T_p - T_r
        s_g = max(0.0, min(1.0, tau_g / T_g))
        if s_g < 0.5:
            env_g = quintic(2.0 * s_g)
        else:
            env_g = quintic(2.0 - 2.0 * s_g)
        glide_feather = math.radians(50.0) * env_g
        joint_pose = dict(pose_t)
        joint_pose["ankle"] = pose_t["ankle"] + glide_feather
        return "glide", joint_pose

class Joint:
    def __init__(self, robot, name, dt_ms):
        self.motor = robot.getDevice(name)
        self.sensor = robot.getDevice(name + "_sensor")
        if self.sensor is not None:
            self.sensor.enable(dt_ms)
        try:
            self.q_min = self.motor.getMinPosition()
            self.q_max = self.motor.getMaxPosition()
        except Exception:
            self.q_min, self.q_max = -3.14, 3.14
        try:
            self.v_max = self.motor.getMaxVelocity()
        except Exception:
            self.v_max = 6.0

    def command(self, q_des, v_des=None):
        q = clamp(q_des, self.q_min + 1e-3, self.q_max - 1e-3)
        if v_des is not None:
            v = clamp(abs(v_des), 0.05, 0.95 * self.v_max)
            try: self.motor.setVelocity(v)
            except Exception: pass
        try:
            self.motor.setPosition(q)
        except Exception: pass

def main():
    p = PARAMS
    dt_ms = p["time_step_ms"]
    dt_s  = dt_ms * 1e-3

    robot = Robot()
    gps = robot.getDevice("gps");   gps.enable(dt_ms)
    imu = robot.getDevice("imu");   imu.enable(dt_ms)
    gyro = robot.getDevice("gyro"); gyro.enable(dt_ms)
    accel = robot.getDevice("accel"); accel.enable(dt_ms)

    JOINT_NAMES_R = ("hipR", "kneeR", "ankleR")
    JOINT_NAMES_L = ("hipL", "kneeL", "ankleL")

    joints = {n: Joint(robot, n, dt_ms) for n in JOINT_NAMES_R + JOINT_NAMES_L}
    pose0 = p["pose_extended"]
    initial = {
        "hipR":   pose0["hip"],   "kneeR":  pose0["knee"],  "ankleR": pose0["ankle"],
        "hipL":   pose0["hip"],   "kneeL":  pose0["knee"],  "ankleL": pose0["ankle"],
    }
    for name, q in initial.items():
        joints[name].command(q, 0.3) 

    yaw_ctrl   = build_controller(CONTROLLER, "yaw",   dt_s)
    depth_ctrl = build_controller(CONTROLLER, "depth", dt_s)
    pitch_ctrl = build_controller(CONTROLLER, "pitch", dt_s)
    roll_ctrl  = build_controller(CONTROLLER, "roll",  dt_s)

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, "frog_log_" + CONTROLLER + ".csv")
    with open(log_path, "w", newline="") as log_f:
        log_w = csv.writer(log_f)
        log_w.writerow([
            "t_s", "controller", "phase",
            "x", "y", "z", "roll", "pitch", "yaw",
            "wx", "wy", "wz",
            "des_yaw", "yaw_err", "u_yaw",
            "depth_err", "u_depth",
            "pitch_err", "u_pitch",
            "roll_err", "u_roll",
            "hipR", "kneeR", "ankleR",
            "hipL", "kneeL", "ankleL",
            "wp_idx", "wp_x", "wp_y", "dist_to_wp",
        ])

        T_cycle = (p["duration_power_s"] + p["duration_glide_s"] + p["duration_recovery_s"])
        print("=" * 60)
        print("frog_controller V13 - RESEARCH-BASED IMPLEMENTATION")
        print("=" * 60)
        print(f"Algorithm: {CONTROLLER}")
        print(f"Joints: 6 (3 per leg, no spokes)")
        print(f"Cycle: {T_cycle:.2f} s (P={p['duration_power_s']:.2f} + "
              f"G={p['duration_glide_s']:.2f} + R={p['duration_recovery_s']:.2f}) = "
              f"{1.0/T_cycle:.2f} Hz")
        print(f"Pose blending: quintic (zero v,a at endpoints)")
        print(f"Phase offsets: knee leads {p['knee_phase_lead_s']*1000:.0f}ms, "
              f"ankle lags {p['ankle_phase_lag_s']*1000:.0f}ms")
        print(f"Target depth: {p['depth_target_m']:.2f} m, "
              f"target pitch: {math.degrees(p['pitch_target_rad']):.0f} deg (nose-up)")
        print("=" * 60)

        t = 0.0
        wp_idx = 0
        waypoints = list(p["waypoints_xy"])
        n_wp = len(waypoints)
        warmup_hold = p["warmup_hold_s"]
        warmup_s    = p["warmup_total_s"]

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

            des_yaw = math.atan2(tgt_y - y, tgt_x - x)
            yaw_err = wrap_to_pi(des_yaw - yaw)
            depth_err = z - p["depth_target_m"]
            pitch_err = pitch - p["pitch_target_rad"]
            roll_err = -roll

            if CONTROLLER == "pid_ff":
                yaw_ctrl.set_feedforward(0.0)
                depth_ctrl.set_feedforward(0.0)
                pitch_ctrl.set_feedforward(0.0)
                roll_ctrl.set_feedforward(0.0)

            u_yaw = yaw_ctrl.step(yaw_err)
            u_depth = depth_ctrl.step(depth_err)
            u_pitch = pitch_ctrl.step(pitch_err)
            u_roll = roll_ctrl.step(roll_err)
            
            soft_start_s = 0.5
            warmup_hold_end = soft_start_s + warmup_hold
            warmup_ramp_end = warmup_hold_end + (warmup_s - warmup_hold)
            
            if t < soft_start_s:
                ramp = quintic(t / soft_start_s)
                neutral = {"hip": 0.0, "knee": 0.0, "ankle": 0.0}
                joint_pose_R = lerp_pose(neutral, p["pose_extended"], ramp)
                joint_pose_L = lerp_pose(neutral, p["pose_extended"], ramp)
                phase = "soft_start"
                vel_scale_R = vel_scale_L = 0.2
            elif t < warmup_hold_end:
                joint_pose_R = dict(p["pose_extended"])
                joint_pose_L = dict(p["pose_extended"])
                phase = "warmup_hold"
                vel_scale_R = vel_scale_L = 0.3
            elif t < warmup_ramp_end:
                dur = max(warmup_ramp_end - warmup_hold_end, 1e-3)
                ramp = quintic((t - warmup_hold_end) / dur)
                _, pose_R_target = gait_state(0.0, p, 'R')
                _, pose_L_target = gait_state(0.0, p, 'L')
                joint_pose_R = lerp_pose(p["pose_extended"], pose_R_target, ramp)
                joint_pose_L = lerp_pose(p["pose_extended"], pose_L_target, ramp)
                phase = "warmup_ramp"
                vel_scale_R = vel_scale_L = 0.15
            else:
                t_cycle = t - warmup_ramp_end
                phase_R, joint_pose_R = gait_state(t_cycle, p, 'R')
                phase_L, joint_pose_L = gait_state(t_cycle, p, 'L')
                phase = phase_R  

                v_max_local = joints["hipR"].v_max
                def _vs(ph):
                    if ph == "power":
                        return 0.90
                    elif ph == "glide":
                        return 0.30 / v_max_local
                    else:  
                        return 1.08 / v_max_local

                vel_scale_R = _vs(phase_R)
                vel_scale_L = _vs(phase_L)

            hip_bias_yaw = 0.0  

            knee_bias_pitch = u_pitch * p["knee_bias_pitch_gain"]
            hip_bias_pitch  = u_pitch * p["hip_bias_pitch_gain"]
            ankle_bias_depth = u_depth * p["ankle_bias_depth_gain"]

            ankle_bias_roll = u_roll * 0.3

            v_hip_R   = vel_scale_R * joints["hipR"].v_max
            v_knee_R  = vel_scale_R * joints["kneeR"].v_max
            v_ankle_R = vel_scale_R * joints["ankleR"].v_max
            v_hip_L   = vel_scale_L * joints["hipL"].v_max
            v_knee_L  = vel_scale_L * joints["kneeL"].v_max
            v_ankle_L = vel_scale_L * joints["ankleL"].v_max

            hipR_cmd  = joint_pose_R["hip"]   + hip_bias_yaw + hip_bias_pitch
            hipL_cmd  = joint_pose_L["hip"]   - hip_bias_yaw + hip_bias_pitch
            kneeR_cmd = joint_pose_R["knee"]  + knee_bias_pitch
            kneeL_cmd = joint_pose_L["knee"]  + knee_bias_pitch
            ankleR_cmd = joint_pose_R["ankle"] + ankle_bias_depth + ankle_bias_roll
            ankleL_cmd = joint_pose_L["ankle"] + ankle_bias_depth - ankle_bias_roll

            joints["hipR"]  .command(hipR_cmd,   v_hip_R)
            joints["kneeR"] .command(kneeR_cmd,  v_knee_R)
            joints["ankleR"].command(ankleR_cmd, v_ankle_R)
            joints["hipL"]  .command(hipL_cmd,   v_hip_L)
            joints["kneeL"] .command(kneeL_cmd,  v_knee_L)
            joints["ankleL"].command(ankleL_cmd, v_ankle_L)
 
            log_w.writerow([
                "%.3f" % t, CONTROLLER, phase,
                x, y, z, roll, pitch, yaw,
                wx, wy, wz,
                des_yaw, yaw_err, u_yaw,
                depth_err, u_depth,
                pitch_err, u_pitch,
                roll_err, u_roll,
                hipR_cmd, kneeR_cmd, ankleR_cmd,
                hipL_cmd, kneeL_cmd, ankleL_cmd,
                wp_idx, wp_x, wp_y, d_to_wp,
            ])


if __name__ == "__main__":
    main()
