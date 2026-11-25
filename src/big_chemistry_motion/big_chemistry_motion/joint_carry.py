from copy import deepcopy
import math
from types import SimpleNamespace

from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectoryPoint

from big_chemistry_motion.cartesian_motion import _call_service
from big_chemistry_motion.motion_goals import build_target_pose


ARM_NAMES = tuple(f'joint{i}' for i in range(1, 7))


def build_joint_candidate(reference_state, goal_positions, *, shoulder_bend_rad=0.0):
    if not math.isfinite(shoulder_bend_rad) or abs(shoulder_bend_rad) > 0.08:
        raise RuntimeError('Shoulder path bend must be finite and within +/-0.08 rad.')
    original = dict(zip(reference_state.joint_state.name,
                        reference_state.joint_state.position))
    if not all(name in original and name in goal_positions for name in ARM_NAMES):
        raise RuntimeError('Joint candidate requires all six arm joints.')
    start = [original[name] for name in ARM_NAMES]
    end = [goal_positions[name] for name in ARM_NAMES]
    if not all(math.isfinite(value) for value in start + end):
        raise RuntimeError('Non-finite joint candidate endpoint.')
    delta = [b-a for a, b in zip(start, end)]
    travel = max(abs(value) for value in delta)
    slope_bound = travel + math.pi * abs(shoulder_bend_rad)
    acceleration_bound = (
        5.774 * slope_bound + abs(shoulder_bend_rad) * math.pi**2 * 1.875**2
    )
    duration = max(2.0, 1.875 * slope_bound / 0.1,
                   math.sqrt(acceleration_bound / 0.05))
    if duration > 80.0:
        raise RuntimeError('Joint candidate exceeds the 80 s duration budget.')
    steps = max(2, math.ceil(duration / 0.1))
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(ARM_NAMES)
    for index in range(steps + 1):
        u = index / steps
        s = 10*u**3 - 15*u**4 + 6*u**5
        ds = (30*u**2 - 60*u**3 + 30*u**4) / duration
        dds = (60*u - 180*u**2 + 120*u**3) / (duration * duration)
        point = JointTrajectoryPoint()
        point.positions = [q + d*s for q, d in zip(start, delta)]
        point.velocities = [d*ds for d in delta]
        point.accelerations = [d*dds for d in delta]
        if shoulder_bend_rad:
            bend = shoulder_bend_rad * math.sin(math.pi*s)
            bend_velocity = shoulder_bend_rad * math.pi * math.cos(math.pi*s) * ds
            bend_acceleration = shoulder_bend_rad * (
                math.pi * math.cos(math.pi*s) * dds
                - math.pi**2 * math.sin(math.pi*s) * ds**2
            )
            for joint_index, sign in ((1, 1.0), (4, -1.0)):
                point.positions[joint_index] += sign * bend
                point.velocities[joint_index] += sign * bend_velocity
                point.accelerations[joint_index] += sign * bend_acceleration
        nanoseconds = round(duration * u * 1e9)
        point.time_from_start.sec = nanoseconds // 1000000000
        point.time_from_start.nanosec = nanoseconds % 1000000000
        trajectory.joint_trajectory.points.append(point)
    # Preserve the exact boundary positions and explicitly zero derivatives.
    trajectory.joint_trajectory.points[0].positions = start
    trajectory.joint_trajectory.points[-1].positions = end
    return trajectory


def _join_stopped_segments(first, second):
    names = first.joint_trajectory.joint_names
    if names != second.joint_trajectory.joint_names:
        raise RuntimeError('Joint-segment names do not match.')
    a = first.joint_trajectory.points[-1]
    b = second.joint_trajectory.points[0]
    if any(abs(x-y) > 1e-10 for x, y in zip(a.positions, b.positions)):
        raise RuntimeError('Joint-segment positions are discontinuous.')
    if any(abs(v) > 1e-10 for p in (a, b) for v in p.velocities + p.accelerations):
        raise RuntimeError('Joint segments must stop at their boundary.')
    trajectory = deepcopy(first)
    offset = a.time_from_start.sec * 1000000000 + a.time_from_start.nanosec
    for source in second.joint_trajectory.points[1:]:
        point = deepcopy(source)
        stamp = offset + source.time_from_start.sec * 1000000000 + source.time_from_start.nanosec
        point.time_from_start.sec = stamp // 1000000000
        point.time_from_start.nanosec = stamp % 1000000000
        trajectory.joint_trajectory.points.append(point)
    return trajectory


def plan_joint_carry_probe(
    node, target, loaded_state, *, via_offset_deg=0.0, shoulder_bend_rad=0.0,
):
    if node.get_parameter('motion_goal').value == 'pickup_fake':
        from big_chemistry_motion.fake_pickup import require_fake_environment
        require_fake_environment(node)
    elif (
        node.get_parameter('motion_goal').value != 'precheck'
        or node.get_parameter('execute').value
    ):
        raise RuntimeError('joint_probe is restricted to preview-only precheck.')
    if not math.isfinite(via_offset_deg) or abs(via_offset_deg) > 20.0:
        raise RuntimeError('Joint-via offset must be finite and within +/-20 degrees.')
    if not math.isfinite(shoulder_bend_rad) or abs(shoulder_bend_rad) > 0.08:
        raise RuntimeError('Shoulder path bend must be finite and within +/-0.08 rad.')
    if {a.object.id for a in loaded_state.attached_collision_objects} != {'plate_base', 'plate_body'}:
        raise RuntimeError('Joint probe requires both attached plate parts.')
    seed_state = deepcopy(loaded_state)
    joints = seed_state.joint_state
    if len(joints.name) != len(joints.position) or len(set(joints.name)) != len(joints.name):
        raise RuntimeError('Invalid joint-probe reference state.')
    positions = dict(zip(joints.name, joints.position))
    if not set(ARM_NAMES).issubset(positions):
        raise RuntimeError('Joint probe is missing arm joints.')
    # A second numerical seed: no stored configuration is used as a motion goal.
    positions.update(joint2=0.3, joint3=1.2, joint4=0.0, joint5=0.9)
    joints.name = list(positions)
    joints.position = list(positions.values())
    joints.velocity = []
    joints.effort = []
    seed_state.is_diff = False
    request = GetPositionIK.Request()
    request.ik_request.group_name = 'lite6'
    request.ik_request.ik_link_name = 'link_eef'
    request.ik_request.robot_state = seed_state
    request.ik_request.avoid_collisions = True
    request.ik_request.pose_stamped.header = deepcopy(target.header)
    request.ik_request.pose_stamped.pose = build_target_pose(target)
    request.ik_request.timeout.sec = 3
    node.get_logger().info(
        'JOINT PROBE: solving the exact exit pose with attached geometry, '
        'collision checking and an alternate IK seed. No execution.'
    )
    response = _call_service(node, GetPositionIK, '/compute_ik', request)
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(f'Loaded exit IK failed: {response.error_code.val}')
    solved = response.solution.joint_state
    if len(solved.name) != len(solved.position) or len(set(solved.name)) != len(solved.name):
        raise RuntimeError('Loaded exit IK returned an invalid joint state.')
    goal = dict(zip(solved.name, solved.position))
    if not all(name in goal and math.isfinite(goal[name]) for name in ARM_NAMES):
        raise RuntimeError('Loaded exit IK returned incomplete arm positions.')
    node.get_logger().info(
        'LOADED EXIT IK [rad]: '
        + ', '.join(f'{name}={goal[name]:.6f}' for name in ARM_NAMES)
    )
    if shoulder_bend_rad:
        node.get_logger().info(
            f'JOINT CARRY SHAPE: shoulder bend={shoulder_bend_rad:.5f} rad; '
            'joint2 and joint5 receive opposite smooth offsets in the exit leg. '
            'Endpoints unchanged; full validation still required.'
        )
    if abs(via_offset_deg) > 1e-12:
        via_state = deepcopy(loaded_state)
        via_positions = dict(zip(via_state.joint_state.name, via_state.joint_state.position))
        offset = math.radians(via_offset_deg)
        for name in ('joint1', 'joint6'):
            via_positions[name] += offset
            if not -3.1101767270538954 <= via_positions[name] <= 3.1101767270538954:
                raise RuntimeError(f'Joint-via candidate exceeds the Lite6 limit: {name}')
        via_state.joint_state.name = list(via_positions)
        via_state.joint_state.position = list(via_positions.values())
        via_state.joint_state.velocity = []
        via_state.joint_state.effort = []
        node.get_logger().info(
            f'JOINT VIA: first sweep joint1 and joint6 together by {via_offset_deg:.2f} deg, '
            'holding other joints. Then connect to the solved exit IK. '
            'Full collision and plate-tilt validation is required.'
        )
        first = build_joint_candidate(loaded_state, via_positions)
        second = build_joint_candidate(
            via_state, goal, shoulder_bend_rad=shoulder_bend_rad,
        )
        trajectory = _join_stopped_segments(first, second)
    else:
        trajectory = build_joint_candidate(
            loaded_state, goal, shoulder_bend_rad=shoulder_bend_rad,
        )
    duration = trajectory.joint_trajectory.points[-1].time_from_start
    node.get_logger().info(
        f'JOINT PROBE: built a {duration.sec + duration.nanosec*1e-9:.3f} s '
        'quintic joint candidate. Collision and plate-tilt validation follows; '
        'this is not an execution command.'
    )
    return SimpleNamespace(start_state=deepcopy(loaded_state), solution=trajectory)
