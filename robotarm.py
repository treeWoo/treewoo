import time     #2026/06/08/15:45
import math
import numpy as np
import keyboard
from dynamixel_sdk import *

ADDR_AX_TORQUE_ENABLE = 24
ADDR_AX_GOAL_POSITION = 30
ADDR_AX_MOVING_SPEED = 32
ADDR_AX_PRESENT_POSITION = 36
LEN_AX_POSITION = 2
PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000
DEVICENAME = 'COM4'

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_ID_LIST = [4, 3, 2]
DIR_MULTIPLIER = [1.0, 1.0, -1.0, 1.0]
OFFSET_DEG = [150.0, 150.0, 150.0, 150.0]
IK_ZERO_DEG = [0.0, 90.0, 0.0, 0.0]

SERVO_MIN_DEG = 0.0
SERVO_MAX_DEG = 300.0

L1 = 0.12
L2 = 0.11
L3 = 0.05
L4 = 0.06
L_pen = 0.097

L34 = L3 + L4
L_VIRTUAL = math.sqrt(L34**2 + L_pen**2)

PAPER_BASE_X_OFFSET_CM = 9.0
PAPER_BASE_Y_OFFSET_CM = 2.0
PAPER_ROT_DEG = 0.0
PAPER_SCALE_X = 1.0
PAPER_SCALE_Y = 1.0

PEN_Z = 0.0097
SMALL_PEN_Z = 0.012
HOVER_Z = 0.06

NORMAL_RADIUS_CM = 3.0
NORMAL_CIRCLE_OVERLAP_DEG = 10.0
SMALL_CIRCLE_OVERLAP_DEG = 10.0
SMALL_LEAD_IN_DEG = 10.0

DT = 0.01
HOVER_DXL_SPEED = 120
DRAW_DXL_SPEED = 130
HOVER_RATE_DEG_S = 35.0
DRAW_RATE_DEG_S = 20.0
LIFT_RATE_DEG_S = 18.0

CIRCLE_POINTS_PER_CM = 10
CIRCLE_MIN_POINTS = 120

PEN_SETTLE_S = 0.08

SMALL_RADIUS_GAIN_X = 1.00
SMALL_RADIUS_GAIN_Y = 1.00
SMALL_CENTER_BIAS_X_CM = -1.2
SMALL_CENTER_BIAS_Y_CM = 0.0

NORMAL_INWARD_COMP_CM = 0.14
SMALL_INWARD_COMP_CM = 0.12
DRAW_COMP_RAMP_RATIO = 0.06

MIN_RECOMMENDED_R_M = 0.075
MIN_HARD_SMALL_R_M = 0.055
REACH_TOL_DXL = 8
REACH_TIMEOUT_S = 3.0


def deg2dxl(deg, motor_idx, strict=True):
    delta_deg = float(deg) - IK_ZERO_DEG[motor_idx]
    target_deg = delta_deg * DIR_MULTIPLIER[motor_idx] + OFFSET_DEG[motor_idx]
    if strict and not (SERVO_MIN_DEG <= target_deg <= SERVO_MAX_DEG):
        raise ValueError(
            f"모터 {DXL_ID_LIST[motor_idx]} 목표각 범위 초과: "
            f"IK={deg:.2f}deg, DXL각={target_deg:.2f}deg"
        )
    target_deg = float(np.clip(target_deg, SERVO_MIN_DEG, SERVO_MAX_DEG))
    return int(round((target_deg / 300.0) * 1023.0))


def set_all_speed(portHandler, packetHandler, speed):
    speed = int(np.clip(speed, 0, 1023))
    for dxl_id in DXL_ID_LIST:
        packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_AX_MOVING_SPEED, speed)


def write_goal_sync(portHandler, packetHandler, th_deg):
    group = GroupSyncWrite(portHandler, packetHandler, ADDR_AX_GOAL_POSITION, LEN_AX_POSITION)
    goals = []
    for motor_idx, dxl_id in enumerate(DXL_ID_LIST):
        pos = deg2dxl(float(th_deg[motor_idx]), motor_idx, strict=True)
        goals.append(pos)
        param = [DXL_LOBYTE(pos), DXL_HIBYTE(pos)]
        if not group.addParam(dxl_id, param):
            raise RuntimeError(f"GroupSyncWrite addParam 실패: ID {dxl_id}")
    result = group.txPacket()
    group.clearParam()
    if result != COMM_SUCCESS:
        raise RuntimeError(packetHandler.getTxRxResult(result))
    return goals


def read_present_positions(portHandler, packetHandler):
    vals = []
    for dxl_id in DXL_ID_LIST:
        pos, comm_result, dxl_error = packetHandler.read2ByteTxRx(
            portHandler, dxl_id, ADDR_AX_PRESENT_POSITION
        )
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(packetHandler.getTxRxResult(comm_result))
        if dxl_error != 0:
            raise RuntimeError(packetHandler.getRxPacketError(dxl_error))
        vals.append(pos)
    return vals


def wait_until_reached(portHandler, packetHandler, goal_dxl, tol=REACH_TOL_DXL, timeout=REACH_TIMEOUT_S):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if keyboard.is_pressed('esc'):
            raise RuntimeError('ESC KEY PRESSED')
        now = read_present_positions(portHandler, packetHandler)
        if max(abs(int(a) - int(b)) for a, b in zip(now, goal_dxl)) <= tol:
            return True
        time.sleep(0.03)
    return False


def smoothstep(s):
    return 3.0 * s * s - 2.0 * s * s * s


def draw_comp_weight(s):
    s = float(np.clip(s, 0.0, 1.0))
    a = float(np.clip(DRAW_COMP_RAMP_RATIO, 0.03, 0.35))
    if s < a:
        return smoothstep(s / a)
    if s > 1.0 - a:
        return smoothstep((1.0 - s) / a)
    return 1.0


def move_segment(th_start, th_end, portHandler, packetHandler, rate_deg_s, profile='linear', wait_end=False):
    th_start = np.asarray(th_start, dtype=float)
    th_end = np.asarray(th_end, dtype=float)
    max_diff = float(np.max(np.abs(th_end - th_start)))
    t_step = max(DT, max_diff / max(rate_deg_s, 1e-6))
    steps = max(1, int(math.ceil(t_step / DT)))
    final_goal = None
    for step in range(1, steps + 1):
        if keyboard.is_pressed('esc'):
            raise RuntimeError('ESC KEY PRESSED')
        s = step / steps
        b = smoothstep(s) if profile == 'cubic' else s
        th = th_start + (th_end - th_start) * b
        final_goal = write_goal_sync(portHandler, packetHandler, th)
        time.sleep(DT)
    final_goal = write_goal_sync(portHandler, packetHandler, th_end)
    if wait_end:
        wait_until_reached(portHandler, packetHandler, final_goal)
    return final_goal


def follow_path(th_waypoints, portHandler, packetHandler, rate_deg_s, profile='linear', wait_last=False):
    th_waypoints = np.asarray(th_waypoints, dtype=float)
    for i in range(len(th_waypoints) - 1):
        move_segment(
            th_waypoints[i],
            th_waypoints[i + 1],
            portHandler,
            packetHandler,
            rate_deg_s=rate_deg_s,
            profile=profile,
            wait_end=(wait_last and i == len(th_waypoints) - 2),
        )


def calculate_IK_bulletproof(target_tip_p, allow_clip=False):
    x, y, z = [float(v) for v in target_tip_p]
    th1 = math.atan2(y, x)
    R = math.sqrt(x**2 + y**2)
    dz = z - L1
    D = math.sqrt(R**2 + dz**2)
    min_D = abs(L2 - L_VIRTUAL) + 1e-4
    max_D = L2 + L_VIRTUAL - 1e-4

    if not (min_D <= D <= max_D):
        if not allow_clip:
            raise ValueError(
                f"작업공간 밖 목표점: x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                f"D={D:.3f}, 허용=[{min_D:.3f}, {max_D:.3f}]"
            )
        D = float(np.clip(D, min_D, max_D))

    gamma = math.atan2(L_pen, L34)
    cos_th3_virtual = (D**2 - L2**2 - L_VIRTUAL**2) / (2.0 * L2 * L_VIRTUAL)
    cos_th3_virtual = float(np.clip(cos_th3_virtual, -1.0, 1.0))
    th3_virtual = -math.acos(cos_th3_virtual)
    th3 = th3_virtual + gamma

    alpha = math.atan2(dz, R)
    cos_beta = (D**2 + L2**2 - L_VIRTUAL**2) / (2.0 * D * L2)
    cos_beta = float(np.clip(cos_beta, -1.0, 1.0))
    beta = math.acos(cos_beta)
    th2 = alpha + beta
    th4 = 0.0

    return np.array([math.degrees(th1), math.degrees(th2), math.degrees(th3), th4], dtype=float)


def paper_to_robot(px_cm, py_cm):
    px = px_cm * PAPER_SCALE_X
    py = py_cm * PAPER_SCALE_Y
    a = math.radians(PAPER_ROT_DEG)
    xr_cm = PAPER_BASE_X_OFFSET_CM + py * math.cos(a) - px * math.sin(a)
    yr_cm = PAPER_BASE_Y_OFFSET_CM + py * math.sin(a) + px * math.cos(a)
    return xr_cm / 100.0, yr_cm / 100.0


def circle_point(px_cm, py_cm, radius_cm, angle_rad, z_height, small_mode=False, comp_cm=0.0):
    if small_mode:
        px_cm = px_cm + SMALL_CENTER_BIAS_X_CM
        py_cm = py_cm + SMALL_CENTER_BIAS_Y_CM
    cx, cy = paper_to_robot(px_cm, py_cm)
    r_eff_m = max(0.0, radius_cm + comp_cm) / 100.0
    gx = SMALL_RADIUS_GAIN_X if small_mode else 1.0
    gy = SMALL_RADIUS_GAIN_Y if small_mode else 1.0
    return [cx + gx * r_eff_m * math.cos(angle_rad), cy + gy * r_eff_m * math.sin(angle_rad), z_height]


def get_normal_circle_waypoints(px_cm, py_cm, radius_cm, z_height):
    n = max(CIRCLE_MIN_POINTS, int(math.ceil(2.0 * math.pi * radius_cm * CIRCLE_POINTS_PER_CM)))
    end_angle = 2.0 * math.pi + math.radians(NORMAL_CIRCLE_OVERLAP_DEG)
    angles = np.linspace(0.0, end_angle, n + 1, endpoint=True)
    pts = []
    comp_vals = []
    for i, a in enumerate(angles):
        s = i / n
        comp = NORMAL_INWARD_COMP_CM * draw_comp_weight(s)
        comp_vals.append(comp)
        pts.append(circle_point(px_cm, py_cm, radius_cm, a, z_height, small_mode=False, comp_cm=comp))
    return pts, max(comp_vals)


def get_small_circle_waypoints(px_cm, py_cm, radius_cm, z_height):
    start_deg = -SMALL_LEAD_IN_DEG
    end_deg = 360.0 + SMALL_CIRCLE_OVERLAP_DEG
    total_deg = end_deg - start_deg
    arc_len_cm = 2.0 * math.pi * radius_cm * (total_deg / 360.0)
    
    n = max(CIRCLE_MIN_POINTS, int(math.ceil(arc_len_cm * CIRCLE_POINTS_PER_CM)))
    
    angles = np.linspace(math.radians(start_deg), math.radians(end_deg), n + 1, endpoint=True)
    pts = []
    comp_vals = []
    for i, a in enumerate(angles):
        s = i / n
        comp = SMALL_INWARD_COMP_CM * draw_comp_weight(s)
        comp_vals.append(comp)
        pts.append(circle_point(px_cm, py_cm, radius_cm, a, z_height, small_mode=True, comp_cm=comp))
    return pts, n, max(comp_vals)


def make_joint_path(points):
    return np.array([calculate_IK_bulletproof(p, allow_clip=False) for p in points], dtype=float)


def unwrap_joint_path(th_waypoints):
    th_waypoints = np.asarray(th_waypoints, dtype=float)
    rad = np.deg2rad(th_waypoints)
    rad[:, :3] = np.unwrap(rad[:, :3], axis=0)
    out = np.rad2deg(rad)
    out[:, 3] = th_waypoints[:, 3]
    return out


def validate_path(points, th_waypoints):
    r_vals = [math.sqrt(p[0] ** 2 + p[1] ** 2) for p in points]
    d_vals = [math.sqrt((math.sqrt(p[0] ** 2 + p[1] ** 2)) ** 2 + (p[2] - L1) ** 2) for p in points]
    for th in th_waypoints:
        for motor_idx in range(4):
            deg2dxl(float(th[motor_idx]), motor_idx, strict=True)
    return min(r_vals), max(r_vals), min(d_vals), max(d_vals)


def main():
    print('==================================================')
    print('  원 2개 연속 그리기 프로그램 - 안쪽 파고듦 보정 및 선형 보간')
    print('==================================================\n')

    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        print('[에러] 통신 포트 연결 실패!')
        return
    if not portHandler.setBaudRate(BAUDRATE):
        print('[에러] Baudrate 설정 실패!')
        portHandler.closePort()
        return

    try:
        set_all_speed(portHandler, packetHandler, HOVER_DXL_SPEED)
        for dxl_id in DXL_ID_LIST:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE)

        th_up = np.array([0.0, 90.0, 0.0, 0.0], dtype=float)
        goal = write_goal_sync(portHandler, packetHandler, th_up)
        wait_until_reached(portHandler, packetHandler, goal)
        print('>>> 로봇이 시작 자세를 잡았습니다.\n')

        while True:
            print('-' * 50)
            user_in = input("새로운 원 2개를 그리시겠습니까? (엔터 입력시 진행 / 종료 'q'): ")
            if user_in.strip().lower() == 'q':
                break

            try:
                print('\n[첫 번째 원 정보]')
                px_1 = float(input('▶ 도면상 X 좌표 [cm]: '))
                py_1 = float(input('▶ 도면상 Y 좌표 [cm]: '))
                r_1 = float(input('▶ 반지름 [cm]: '))
                if r_1 <= 0:
                    raise ValueError('반지름은 0보다 커야 합니다.')

                print('\n[두 번째 원 정보]')
                px_2 = float(input('▶ 도면상 X 좌표 [cm]: '))
                py_2 = float(input('▶ 도면상 Y 좌표 [cm]: '))
                r_2 = float(input('▶ 반지름 [cm]: '))
                if r_2 <= 0:
                    raise ValueError('반지름은 0보다 커야 합니다.')
                
                circles = [(px_1, py_1, r_1), (px_2, py_2, r_2)]
            except ValueError as e:
                print(f'\n[경고] 숫자를 정확히 입력해주세요. {e}')
                continue

            try:
                # 시작 지점은 위를 보고 있는 초기 상태 (th_up)
                current_th = th_up

                for i, (px_in, py_in, r_in) in enumerate(circles):
                    small_mode = r_in < NORMAL_RADIUS_CM
                    if small_mode:
                        z_draw = SMALL_PEN_Z
                        circle_pts, steps, max_comp = get_small_circle_waypoints(px_in, py_in, r_in, z_draw)
                        mode_msg = '작은 원 보정 모드'
                    else:
                        z_draw = PEN_Z
                        circle_pts, max_comp = get_normal_circle_waypoints(px_in, py_in, r_in, z_draw)
                        steps = len(circle_pts) - 1
                        mode_msg = '큰 원/일반 원 모드'

                    start_p = circle_pts[0]
                    end_p = circle_pts[-1]
                    hover_start = [start_p[0], start_p[1], HOVER_Z]
                    hover_end = [end_p[0], end_p[1], HOVER_Z]

                    th_hover_start = calculate_IK_bulletproof(hover_start)
                    th_circle = unwrap_joint_path(make_joint_path(circle_pts))
                    th_hover_end = calculate_IK_bulletproof(hover_end)

                    min_R, max_R, min_D, max_D = validate_path(circle_pts, th_circle)
                    
                    circle_num = '첫 번째' if i == 0 else '두 번째'
                    print(f'\n[{circle_num} 원 진단] {mode_msg}')
                    print(f'[{circle_num} 원 진단] R 범위={min_R*100:.1f}~{max_R*100:.1f}cm, D 범위={min_D*100:.1f}~{max_D*100:.1f}cm')
                    print(f'[{circle_num} 원 진단] 안쪽 파고듦 보정: 최대 +{max_comp:.2f}cm')
                    
                    if small_mode:
                        if min_R < MIN_HARD_SMALL_R_M:
                            print('[강한 주의] 원이 베이스 축에 너무 가깝습니다. 찌그러짐이 남을 수 있습니다.')
                        elif min_R < MIN_RECOMMENDED_R_M:
                            print('[주의] 베이스 축에 가까운 작은 원입니다. 종이를 조금 이동하면 정확도가 좋아집니다.')

                    print(f'[{circle_num} 원 작업 시작] P({px_in}, {py_in}), 반경 {r_in}cm 원을 그립니다.')

                    # 1. 호버링 위치로 이동 (이전 위치에서 이동)
                    set_all_speed(portHandler, packetHandler, HOVER_DXL_SPEED)
                    move_segment(current_th, th_hover_start, portHandler, packetHandler, HOVER_RATE_DEG_S, profile='cubic', wait_end=True)

                    # 2. 펜 내리기
                    set_all_speed(portHandler, packetHandler, DRAW_DXL_SPEED)
                    move_segment(th_hover_start, th_circle[0], portHandler, packetHandler, LIFT_RATE_DEG_S, profile='cubic', wait_end=True)

                    if small_mode:
                        time.sleep(PEN_SETTLE_S)

                    # 3. 원 그리기
                    follow_path(th_circle, portHandler, packetHandler, DRAW_RATE_DEG_S, profile='linear', wait_last=False)

                    # 4. 펜 들어올리기
                    set_all_speed(portHandler, packetHandler, HOVER_DXL_SPEED)
                    move_segment(th_circle[-1], th_hover_end, portHandler, packetHandler, LIFT_RATE_DEG_S, profile='cubic', wait_end=True)

                    # 다음 원을 위해 현재 위치 업데이트
                    current_th = th_hover_end

                # 모든 원을 다 그린 후 시작 자세로 복귀
                move_segment(current_th, th_up, portHandler, packetHandler, HOVER_RATE_DEG_S, profile='cubic', wait_end=True)
                print('\n[작업 완료] 2개의 원 그리기 완료 후 시작 자세로 복귀했습니다.\n')

            except ValueError as e:
                print(f'\n[경고] {e}')
                print('입력 좌표/반지름 또는 오프셋을 조정한 뒤 다시 시도하세요.\n')

    except RuntimeError as e:
        if 'ESC' in str(e):
            print('\n[긴급 정지] ESC 키가 눌려 동작이 중단되었습니다.')
        else:
            print(f'\n[에러] {e}')
    except KeyboardInterrupt:
        print('\n[긴급 정지] 터미널 강제 종료가 감지되었습니다.')
    finally:
        print('>>> 모든 모터의 토크를 해제하고 포트를 정지합니다. <<<')
        for dxl_id in DXL_ID_LIST:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE)
        portHandler.closePort()

if __name__ == '__main__':
    main()