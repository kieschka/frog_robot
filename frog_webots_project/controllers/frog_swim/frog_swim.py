from controller import Supervisor
import math
import csv
import os

TIME_STEP = 16

# ---------------------------
# Основные настраиваемые параметры
# ---------------------------

# Размеры бассейна в плоскости XZ
POOL_HALF_X = 7.8
POOL_HALF_Z = 10.0
SAFE_MARGIN = 1.6

# Смещение контрольной точки в носовой части корпуса
NOSE_OFFSET = 0.42

# Параметры движения у стенки
TURN_TRIGGER = 1.05       # раньше начинать маневр у стены
HARD_CONTACT = 0.42       # очень близкий контакт
ESCAPE_RELEASE = 1.85     # когда можно перейти из escape в pivot
PIVOT_EXIT_ANGLE = 0.24   # когда считать поворот завершенным
PIVOT_MAX_TIME = 2.4      # защита, чтобы не крутиться бесконечно
ESCAPE_MIN_TIME = 0.55    # минимум времени держать escape

# Параметры самовыравнивания
SELF_RIGHT_ROLL = 1.15    # ~66 deg
SELF_RIGHT_PITCH = 1.00   # ~57 deg
SELF_RIGHT_TIME = 1.10
SELF_RIGHT_KP = 95.0
SELF_RIGHT_KD = 16.0
SELF_RIGHT_MAX_TORQUE = 52.0
SELF_RIGHT_UP_FORCE = 70.0

# Стабилизация глубины и положения корпуса
TARGET_Y = 0.72
DEPTH_KP = 92.0
DEPTH_KD = 28.0
MAX_VERTICAL_FORCE = 110.0

ATTITUDE_KP = 38.0
ATTITUDE_KD = 8.0
MAX_ATTITUDE_TORQUE = 20.0

# Управление поворотом в горизонтальной плоскости
CRUISE_YAW_KP = 9.0
CRUISE_YAW_KD = 2.8
CRUISE_MAX_YAW_TORQUE = 8.0

PIVOT_YAW_KP = 44.0
PIVOT_YAW_KD = 7.0
PIVOT_MAX_YAW_TORQUE = 26.0

# Силы отхода от стенки
ESCAPE_REVERSE_FORCE = 86.0     # назад вдоль корпуса
ESCAPE_INWARD_FORCE = 92.0      # внутрь бассейна по нормали к стене
CONTACT_NORMAL_FORCE = 120.0    # если уже почти упёрлась
CONTACT_FORWARD_BLOCK = 130.0    # гасит движение в стену
CONTACT_TANGENT_DAMPING = 72.0  # гасит ползание вдоль стены

# Тяга и сопротивление
BASE_THRUST = 34.0
POWER_THRUST_GAIN = 44.0
MAX_FORWARD_FORCE = 74.0
DRAG_XZ = 13.0
DRAG_Y = 18.0

# Параметры цикла гребка
CYCLE_TIME = 1.35
POWER_PORTION = 0.38
PHASE_OFFSET_RIGHT = 0.50

POWER_SPEED = 1.0
RECOVERY_SPEED = 0.56
JOINT_CMD_SPEED = {"hip": 3.0, "knee": 4.1, "ankle": 4.9}

LEFT_RECOVERY = {"hip": 0.58, "knee": -1.22, "ankle": 0.92}
LEFT_POWER    = {"hip": -0.36, "knee": -0.26, "ankle": 0.26}

PALM_MIN = 0.45
PALM_MAX = 1.0

WAYPOINTS = [
    (-4.8, -5.8),
    ( 0.0, -3.2),
    ( 4.8, -5.8),
    ( 5.8,  0.0),
    ( 4.8,  5.8),
    ( 0.0,  3.2),
    (-4.8,  5.8),
    (-5.8,  0.0),
]

# ---------------------------
# Вспомогательные функции
# ---------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def normalize_xz(vx, vz, fallback=(0.0, 1.0)):
    n = math.sqrt(vx * vx + vz * vz)
    if n < 1e-9:
        return fallback
    return vx / n, vz / n

def smoothstep(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def interp_pose(a, b, s):
    return {k: a[k] + (b[k] - a[k]) * s for k in a}

def right_from_left(left_pose):
    return {
        "hip": -left_pose["hip"],
        "knee": -left_pose["knee"],
        "ankle": -left_pose["ankle"],
    }

def heading_to_dir(yaw):
    return math.sin(yaw), math.cos(yaw)

def rot_axis_angle_to_rpy(axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0, 0.0, 0.0
    x /= n
    y /= n
    z /= n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    r00 = c + x * x * C
    r10 = y * x * C + z * s
    r20 = z * x * C - y * s
    r21 = z * y * C + x * s
    r22 = c + z * z * C

    pitch = math.asin(clamp(-r20, -1.0, 1.0))
    roll = math.atan2(r21, r22)
    yaw = math.atan2(r10, r00)
    return roll, pitch, yaw

def piecewise_phase(phi):
    if phi < POWER_PORTION:
        return phi / max(POWER_PORTION, 1e-6), 1.0
    return (phi - POWER_PORTION) / max(1.0 - POWER_PORTION, 1e-6), 0.0

def phase_targets(sim_time, offset):
    phi = ((sim_time / CYCLE_TIME) + offset) % 1.0
    local_s, in_power = piecewise_phase(phi)
    s = smoothstep(local_s)

    if in_power > 0.5:
        pose = interp_pose(LEFT_RECOVERY, LEFT_POWER, s)
        palm = PALM_MIN + (PALM_MAX - PALM_MIN) * s
        speed_scale = POWER_SPEED
        effort = s
    else:
        pose = interp_pose(LEFT_POWER, LEFT_RECOVERY, s)
        palm = PALM_MAX + (PALM_MIN - PALM_MAX) * s
        speed_scale = RECOVERY_SPEED
        effort = 0.16 * (1.0 - s)

    return pose, palm, speed_scale, effort

def choose_waypoint(x, z, current_index):
    wx, wz = WAYPOINTS[current_index]
    if math.hypot(wx - x, wz - z) < 1.5:
        current_index = (current_index + 1) % len(WAYPOINTS)
        wx, wz = WAYPOINTS[current_index]

    if abs(x) > (POOL_HALF_X - SAFE_MARGIN) or abs(z) > (POOL_HALF_Z - SAFE_MARGIN):
        best_i = 0
        best_d = 1e18
        for i, (cx, cz) in enumerate(WAYPOINTS):
            d = (cx - x) ** 2 + (cz - z) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        current_index = best_i
        wx, wz = WAYPOINTS[current_index]
    return current_index, wx, wz

def wall_clearances_from_nose(nx, nz):
    return {
        "east":  POOL_HALF_X - nx,
        "west":  nx + POOL_HALF_X,
        "north": POOL_HALF_Z - nz,
        "south": nz + POOL_HALF_Z,
    }

def nearest_wall(nx, nz):
    cl = wall_clearances_from_nose(nx, nz)
    wall = min(cl, key=cl.get)
    return wall, cl[wall]

def inward_normal(wall):
    return {
        "north": (0.0, -1.0),
        "south": (0.0,  1.0),
        "east":  (-1.0, 0.0),
        "west":  (1.0,  0.0),
    }[wall]

def tangent_for_turn(wall, sign):
    if wall in ("north", "south"):
        return (1.0, 0.0) if sign > 0 else (-1.0, 0.0)
    return (0.0, 1.0) if sign > 0 else (0.0, -1.0)

# ---------------------------
# Обёртка для работы с суставами
# ---------------------------

class Joint:
    def __init__(self, motor, sensor):
        self.motor = motor
        self.sensor = sensor
        self.sensor.enable(TIME_STEP)

        try:
            self.min_pos = motor.getMinPosition()
            self.max_pos = motor.getMaxPosition()
            if self.min_pos >= self.max_pos:
                raise ValueError("invalid limits")
        except Exception:
            self.min_pos = -10.0
            self.max_pos = 10.0

        try:
            self.max_vel = motor.getMaxVelocity()
            if self.max_vel <= 0.0:
                raise ValueError("invalid max velocity")
        except Exception:
            self.max_vel = 10.0

    def command(self, q_des, speed):
        q_des = clamp(q_des, self.min_pos + 1e-4, self.max_pos - 1e-4)
        safe_speed = clamp(abs(speed), 0.05, 0.92 * self.max_vel)
        self.motor.setVelocity(safe_speed)
        self.motor.setPosition(q_des)

# ---------------------------
# 4) Компоненты
# ---------------------------

robot = Supervisor()
frog = robot.getSelf()

def dev(name):
    return robot.getDevice(name)

joints = {
    "left_hip":   Joint(dev("left_hip"),   dev("left_hip_sensor")),
    "left_knee":  Joint(dev("left_knee"),  dev("left_knee_sensor")),
    "left_ankle": Joint(dev("left_ankle"), dev("left_ankle_sensor")),
    "right_hip":   Joint(dev("right_hip"),   dev("right_hip_sensor")),
    "right_knee":  Joint(dev("right_knee"),  dev("right_knee_sensor")),
    "right_ankle": Joint(dev("right_ankle"), dev("right_ankle_sensor")),
}

left_palm_scale_node = robot.getFromDef("LEFT_PALM_SCALE")
right_palm_scale_node = robot.getFromDef("RIGHT_PALM_SCALE")
left_scale_field = left_palm_scale_node.getField("scale") if left_palm_scale_node else None
right_scale_field = right_palm_scale_node.getField("scale") if right_palm_scale_node else None

controller_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(controller_dir, "frog_swim_log.csv")
path_path = os.path.join(controller_dir, "frog_swim_path.csv")

log_file = open(log_path, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "t_s", "mode", "wall",
    "x_m", "y_m", "z_m",
    "nose_x", "nose_z",
    "roll_rad", "pitch_rad", "heading_rad",
    "desired_heading_rad", "heading_error_rad",
    "clearance_m",
    "vx_mps", "vy_mps", "vz_mps",
    "wx_radps", "wy_radps", "wz_radps",
    "left_effort", "right_effort",
    "wp_x", "wp_z"
])

path_file = open(path_path, "w", newline="")
path_writer = csv.writer(path_file)
path_writer.writerow(["t_s", "x_m", "y_m", "z_m"])

print("frog_swim v35 wall escape + self-righting started")

# ---------------------------
# 5) Состояния
# ---------------------------

sim_time = 0.0
waypoint_index = 0

nav_mode = "cruise"    # cruise / escape / pivot / self_right
current_wall = "none"
turn_sign = 1.0
pivot_heading = None
mode_start_time = 0.0

# ---------------------------
# 6) Main
# ---------------------------

while robot.step(TIME_STEP) != -1:
    dt = TIME_STEP / 1000.0
    sim_time += dt

    x, y, z = frog.getPosition()
    vx, vy, vz, wx, wy, wz_ang = frog.getVelocity()

    rot = frog.getField("rotation").getSFRotation()
    roll, pitch, yaw = rot_axis_angle_to_rpy(rot[:3], rot[3])

    fwd_x, fwd_z = heading_to_dir(yaw)
    nose_x = x + NOSE_OFFSET * fwd_x
    nose_z = z + NOSE_OFFSET * fwd_z

    wall, clearance = nearest_wall(nose_x, nose_z)

    waypoint_index, wp_x, wp_z = choose_waypoint(x, z, waypoint_index)
    des_x, des_z = normalize_xz(wp_x - x, wp_z - z, (0.0, 1.0))
    desired_heading = math.atan2(des_x, des_z)

    # --- self-right detection first ---
    heavily_tilted = (abs(roll) > SELF_RIGHT_ROLL) or (abs(pitch) > SELF_RIGHT_PITCH)
    if heavily_tilted and nav_mode != "self_right":
        nav_mode = "self_right"
        mode_start_time = sim_time

    # --- переходы между состояниями ---
    if nav_mode == "cruise":
        current_wall = wall
        pivot_heading = None

        if clearance < TURN_TRIGGER:
            nav_mode = "escape"
            current_wall = wall
            mode_start_time = sim_time

            if wall in ("north", "south"):
                turn_sign = 1.0 if x < 0.0 else -1.0
            else:
                turn_sign = 1.0 if z > 0.0 else -1.0

    elif nav_mode == "escape":
        current_wall = wall
        if (sim_time - mode_start_time) > ESCAPE_MIN_TIME and clearance > ESCAPE_RELEASE:
            nav_mode = "pivot"
            mode_start_time = sim_time
            nx, nz = inward_normal(current_wall)
            tx, tz = tangent_for_turn(current_wall, turn_sign)
            dx, dz = normalize_xz(nx + 1.15 * tx, nz + 1.15 * tz)
            pivot_heading = math.atan2(dx, dz)

    elif nav_mode == "pivot":
        target_heading = pivot_heading if pivot_heading is not None else desired_heading
        heading_error = wrap_to_pi(target_heading - yaw)
        if (abs(heading_error) < PIVOT_EXIT_ANGLE and clearance > 1.10) or ((sim_time - mode_start_time) > PIVOT_MAX_TIME):
            nav_mode = "cruise"
            mode_start_time = sim_time
            current_wall = "none"
            pivot_heading = None

    elif nav_mode == "self_right":
        # keep self-righting until stabilized
        if (abs(roll) < 0.45 and abs(pitch) < 0.45 and (sim_time - mode_start_time) > 0.35) or ((sim_time - mode_start_time) > SELF_RIGHT_TIME):
            nav_mode = "cruise"
            mode_start_time = sim_time

    left_pose, left_palm, left_speed_scale, left_effort = phase_targets(sim_time, 0.0)
    right_pose_left_space, right_palm, right_speed_scale, right_effort = phase_targets(sim_time, PHASE_OFFSET_RIGHT)
    right_pose = right_from_left(right_pose_left_space)

    if nav_mode == "self_right":
        left_effort *= 0.15
        right_effort *= 0.15
        left_speed_scale *= 0.65
        right_speed_scale *= 0.65

    for joint_name in ("hip", "knee", "ankle"):
        joints[f"left_{joint_name}"].command(
            left_pose[joint_name],
            JOINT_CMD_SPEED[joint_name] * left_speed_scale
        )
        joints[f"right_{joint_name}"].command(
            right_pose[joint_name],
            JOINT_CMD_SPEED[joint_name] * right_speed_scale
        )

    if left_scale_field:
        left_scale_field.setSFVec3f([1.0, 1.0, left_palm])
    if right_scale_field:
        right_scale_field.setSFVec3f([1.0, 1.0, right_palm])

    leg_effort = 0.5 * (left_effort + right_effort)
    forward_force_mag = clamp(BASE_THRUST + POWER_THRUST_GAIN * leg_effort, 0.0, MAX_FORWARD_FORCE)

    fx = 0.0
    fy = 0.0
    fz = 0.0
    yaw_torque = 0.0

    # --- состояния ---
    if nav_mode == "cruise":
        fx += forward_force_mag * des_x
        fz += forward_force_mag * des_z
        heading_error = wrap_to_pi(desired_heading - yaw)
        yaw_torque = clamp(
            CRUISE_YAW_KP * heading_error - CRUISE_YAW_KD * wy,
            -CRUISE_MAX_YAW_TORQUE,
            CRUISE_MAX_YAW_TORQUE
        )

    elif nav_mode == "escape":
        nx, nz = inward_normal(current_wall)

        # strong push away from wall and backward along body
        fx += -ESCAPE_REVERSE_FORCE * fwd_x + ESCAPE_INWARD_FORCE * nx
        fz += -ESCAPE_REVERSE_FORCE * fwd_z + ESCAPE_INWARD_FORCE * nz

        if current_wall in ("north", "south"):
            fx += -CONTACT_TANGENT_DAMPING * vx
        else:
            fz += -CONTACT_TANGENT_DAMPING * vz

        yaw_torque = 0.0

        if clearance < HARD_CONTACT:
            fx += CONTACT_NORMAL_FORCE * nx
            fz += CONTACT_NORMAL_FORCE * nz
            into_wall_speed = -(vx * nx + vz * nz)
            if into_wall_speed > 0.0:
                fx += CONTACT_FORWARD_BLOCK * nx
                fz += CONTACT_FORWARD_BLOCK * nz

    elif nav_mode == "pivot":
        # minimal inward drift to avoid sticking again
        nx, nz = inward_normal(current_wall)
        fx += 0.16 * forward_force_mag * nx
        fz += 0.16 * forward_force_mag * nz

        target_heading = pivot_heading if pivot_heading is not None else desired_heading
        heading_error = wrap_to_pi(target_heading - yaw)

        yaw_torque = clamp(
            PIVOT_YAW_KP * heading_error - PIVOT_YAW_KD * wy,
            -PIVOT_MAX_YAW_TORQUE,
            PIVOT_MAX_YAW_TORQUE
        )

    else: 
        yaw_torque = 0.0
        fy += SELF_RIGHT_UP_FORCE

    fx += -DRAG_XZ * vx
    fz += -DRAG_XZ * vz
    fy += -DRAG_Y * vy

    fy += clamp(
        DEPTH_KP * (TARGET_Y - y) - DEPTH_KD * vy,
        -MAX_VERTICAL_FORCE,
        MAX_VERTICAL_FORCE
    )

    if nav_mode == "self_right":
        roll_torque = clamp(
            -SELF_RIGHT_KP * roll - SELF_RIGHT_KD * wx,
            -SELF_RIGHT_MAX_TORQUE,
            SELF_RIGHT_MAX_TORQUE
        )
        pitch_torque = clamp(
            -SELF_RIGHT_KP * pitch - SELF_RIGHT_KD * wz_ang,
            -SELF_RIGHT_MAX_TORQUE,
            SELF_RIGHT_MAX_TORQUE
        )
    else:
        roll_torque = clamp(
            -ATTITUDE_KP * roll - ATTITUDE_KD * wx,
            -MAX_ATTITUDE_TORQUE,
            MAX_ATTITUDE_TORQUE
        )
        pitch_torque = clamp(
            -ATTITUDE_KP * pitch - ATTITUDE_KD * wz_ang,
            -MAX_ATTITUDE_TORQUE,
            MAX_ATTITUDE_TORQUE
        )

    frog.addForce([fx, fy, fz], False)
    frog.addTorque([roll_torque, yaw_torque, pitch_torque], False)

    target_heading = desired_heading if nav_mode != "pivot" else (pivot_heading if pivot_heading is not None else desired_heading)
    heading_error = wrap_to_pi(target_heading - yaw)

    log_writer.writerow([
        round(sim_time, 3), nav_mode, current_wall,
        x, y, z,
        nose_x, nose_z,
        roll, pitch, yaw,
        target_heading, heading_error,
        clearance,
        vx, vy, vz,
        wx, wy, wz_ang,
        left_effort, right_effort,
        wp_x, wp_z
    ])
    path_writer.writerow([round(sim_time, 3), x, y, z])

log_file.close()
path_file.close()
