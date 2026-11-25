from copy import deepcopy
import math
from types import SimpleNamespace

import rclpy
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionFK

from big_chemistry_motion.cartesian_motion import (
    _quaternion_multiply, _quaternion_tuple, state_at_trajectory_point,
)
from big_chemistry_motion.motion_checks import diagnose_cartesian_collisions


def _seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def _interpolate_positions(a, b, duration, u):
    result = []
    for i, (q0, q1) in enumerate(zip(a.positions, b.positions)):
        delta = q1 - q0
        if not a.velocities:
            value = q0 + u * delta
        elif not a.accelerations:
            v0 = duration * a.velocities[i]
            v1 = duration * b.velocities[i]
            value = q0 + u * (v0 + u * (
                3.0 * delta - 2.0 * v0 - v1
                + u * (-2.0 * delta + v0 + v1)
            ))
        else:
            c0 = q0
            c1 = duration * a.velocities[i]
            c2 = 0.5 * duration * duration * a.accelerations[i]
            d = q1 - c0 - c1 - c2
            v = duration * b.velocities[i] - c1 - 2.0 * c2
            acc = duration * duration * b.accelerations[i] - 2.0 * c2
            c3 = 10.0 * d - 4.0 * v + 0.5 * acc
            c4 = -15.0 * d + 7.0 * v - acc
            c5 = 6.0 * d - 3.0 * v + 0.5 * acc
            value = c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5))))
        result.append(value)
    return result


def densify_trajectory(trajectory, *, max_dt=0.02, max_joint_step=0.01):
    if max_dt <= 0.0 or max_joint_step <= 0.0:
        raise RuntimeError('Sampling bounds must be positive.')
    source = trajectory.joint_trajectory
    n = len(source.joint_names)
    if not n or len(source.points) < 2 or len(set(source.joint_names)) != n:
        raise RuntimeError('Loaded validation requires a nonempty, timed joint path.')

    modes = set()
    for point in source.points:
        for field in ('positions', 'velocities', 'accelerations'):
            values = getattr(point, field)
            if (field == 'positions' and len(values) != n) or (
                field != 'positions' and len(values) not in (0, n)
            ):
                raise RuntimeError(f'Incomplete trajectory {field}.')
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f'Non-finite trajectory {field}.')
        if point.accelerations and not point.velocities:
            raise RuntimeError('Accelerations without velocities are unsupported.')
        modes.add((bool(point.velocities), bool(point.accelerations)))
    if len(modes) != 1:
        raise RuntimeError('Inconsistent interpolation data between trajectory points.')
    if _seconds(source.points[0]) < 0.0:
        raise RuntimeError('Negative trajectory time.')

    result = deepcopy(trajectory)
    points = [deepcopy(source.points[0])]
    for a, b in zip(source.points, source.points[1:]):
        duration = _seconds(b) - _seconds(a)
        if duration <= 0.0 or not math.isfinite(duration):
            raise RuntimeError('Trajectory timestamps must strictly increase.')
        delta = max(abs(q1 - q0) for q0, q1 in zip(a.positions, b.positions))
        if delta > 0.25:
            raise RuntimeError('Large joint discontinuity in loaded path (>0.25 rad).')
        steps = max(1, math.ceil(duration / max_dt), math.ceil(delta / max_joint_step))
        if len(points) + steps > 5000:
            raise RuntimeError('Loaded path exceeds the 5000-sample validation budget.')
        for step in range(1, steps + 1):
            if step == steps:
                point = deepcopy(b)
            else:
                point = deepcopy(a)
                point.positions = _interpolate_positions(a, b, duration, step / steps)
                if not all(math.isfinite(value) for value in point.positions):
                    raise RuntimeError('Non-finite interpolated joint state.')
                point.velocities = []
                point.accelerations = []
                point.effort = []
                nanoseconds = round((_seconds(a) + duration * step / steps) * 1e9)
                point.time_from_start.sec = nanoseconds // 1000000000
                point.time_from_start.nanosec = nanoseconds % 1000000000
            points.append(point)
    result.joint_trajectory.points = points
    return result


def plate_tilt_degrees(tool_orientation, object_orientation, primitive_orientation):
    q = _quaternion_multiply(
        _quaternion_multiply(_quaternion_tuple(tool_orientation),
                             _quaternion_tuple(object_orientation)),
        _quaternion_tuple(primitive_orientation),
    )
    norm = math.sqrt(sum(v * v for v in q))
    x, y, _, _ = (v / norm for v in q)
    normal_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, normal_z))))


def validate_loaded_trajectory(
    node, response, reference_state, *, label, allow_initial_support=False,
    max_tilt_deg=0.5,
):
    if not math.isfinite(max_tilt_deg) or max_tilt_deg <= 0.0:
        raise RuntimeError('Plate tilt tolerance must be finite and positive.')
    attachments = {a.object.id: a for a in reference_state.attached_collision_objects}
    if set(attachments) != {'plate_base', 'plate_body'}:
        raise RuntimeError('Loaded validation requires both attached plate parts.')
    body = attachments['plate_body']
    if (
        body.link_name != 'link_eef'
        or body.object.header.frame_id != 'link_eef'
        or len(body.object.primitives) != 1
        or len(body.object.primitive_poses) != 1
    ):
        raise RuntimeError('Expected one plate-body primitive expressed in link_eef.')

    dense = densify_trajectory(response.solution)
    names = dense.joint_trajectory.joint_names
    source_joints = reference_state.joint_state
    original = dict(zip(source_joints.name, source_joints.position))
    first = dense.joint_trajectory.points[0]
    if any(name not in original or not math.isfinite(original[name])
           or abs(original[name] - position) > 0.0001
           for name, position in zip(names, first.positions)):
        raise RuntimeError(f'{label}: trajectory does not start at the supplied state.')
    dense_response = SimpleNamespace(
        start_state=state_at_trajectory_point(reference_state, dense, point_index=0),
        solution=dense,
    )
    count = len(dense.joint_trajectory.points)
    node.get_logger().info(
        f'{label}: validating {count} spline samples (<=20 ms spacing); '
        f'plate tilt limit={max_tilt_deg:.2f} deg.'
    )
    if not diagnose_cartesian_collisions(
        node, dense_response, allow_initial_support=allow_initial_support,
    ):
        raise RuntimeError(f'{label}: dense collision validation failed.')

    client = node.create_client(GetPositionFK, '/compute_fk')
    maximum = 0.0
    try:
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('FK service unavailable for plate-tilt validation.')
        for index in range(count):
            state = state_at_trajectory_point(reference_state, dense, point_index=index)
            request = GetPositionFK.Request()
            request.header.frame_id = 'world'
            request.fk_link_names = ['link_eef']
            request.robot_state = state
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
            if not future.done():
                future.cancel()
                raise RuntimeError('Plate-tilt FK timed out.')
            result = future.result()
            if result is None or result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise RuntimeError('Plate-tilt FK failed.')
            if len(result.pose_stamped) != 1 or result.pose_stamped[0].header.frame_id != 'world':
                raise RuntimeError('Unexpected plate-tilt FK reference frame.')
            tilt = plate_tilt_degrees(
                result.pose_stamped[0].pose.orientation,
                body.object.pose.orientation,
                body.object.primitive_poses[0].orientation,
            )
            maximum = max(maximum, tilt)
            if tilt > max_tilt_deg:
                raise RuntimeError(
                    f'{label}: plate tilt={tilt:.5f} deg at sample {index + 1}/{count}.'
                )
    finally:
        node.destroy_client(client)
    node.get_logger().info(
        f'{label} VALIDATED: {count} sampled states collision-free '
        f'(initial support contact only if explicitly allowed); '
        f'maximum plate tilt={maximum:.6f} deg.'
    )

def check_trajectory_limits(result, limits):
    trajectory = result.trajectory
    names = trajectory.joint_trajectory.joint_names
    arm_names = {f'joint{i}' for i in range(1, 7)}

    if len(names) != 6 or set(names) != arm_names:
        raise RuntimeError('Fake pickup paths must contain exactly the six arm joints.')
    if trajectory.multi_dof_joint_trajectory.points:
        raise RuntimeError('Unexpected multi-DOF trajectory.')
    points = trajectory.joint_trajectory.points
    if abs(points[0].time_from_start.sec + points[0].time_from_start.nanosec*1e-9) > 1e-9:
        raise RuntimeError('Trajectory must start at time zero.')
    for point in (points[0], points[-1]):
        if not point.velocities or any(abs(v) > 1e-5 for v in point.velocities):
            raise RuntimeError('Each pickup segment must start and finish at rest.')
    dense = densify_trajectory(trajectory)
    for point in dense.joint_trajectory.points:
        for name, position in zip(names, point.positions):
            lower, upper = limits[name]
            if not lower <= position <= upper:
                raise RuntimeError(f'Sampled spline exceeds URDF joint limits: {name}')
    return dense

def validate_unloaded_trajectory(node, result, limits, *, label):
    if result.trajectory_start.attached_collision_objects:
        raise RuntimeError('Unloaded path unexpectedly has attachments.')

    dense = check_trajectory_limits(result, limits)

    response = SimpleNamespace(
        start_state=result.trajectory_start,
        solution=dense,
    )

    if not diagnose_cartesian_collisions(
        node,
        response,
        allow_initial_support=False,
    ):
        raise RuntimeError(
            f'{label}: dense spline collision validation failed.'
        )

    node.get_logger().info(
        f'{label}: dense spline collision validation passed.'
    )