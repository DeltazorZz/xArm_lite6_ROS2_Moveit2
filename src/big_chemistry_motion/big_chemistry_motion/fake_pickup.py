from copy import deepcopy
import math
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import rclpy

from controller_manager_msgs.srv import ListHardwareComponents
from moveit_msgs.msg import CollisionObject
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters

from big_chemistry_motion.cartesian_motion import (
    _call_service, build_hypothetical_grasp, state_at_trajectory_point,
)
from big_chemistry_motion.gripper_control import close_gripper
from big_chemistry_motion.loaded_validation import (
    densify_trajectory, validate_loaded_trajectory,
)
from big_chemistry_motion.motion_checks import (
    check_execution_start, check_planning_ready, diagnose_cartesian_collisions,
)
from big_chemistry_motion.precheck_scene import PLATE_IDS, _same_object_geometry
from big_chemistry_motion.trajectory_execution import execute_reviewed_trajectory
from big_chemistry_motion.trajectory_tools import show_trajectory
from big_chemistry_motion.hardware_environment import (
    read_arm_joint_limits,
    require_hardware_environment,
)
from controller_manager_msgs.srv import ListHardwareComponents

from big_chemistry_motion.loaded_validation import (
    check_trajectory_limits,
    densify_trajectory,
    validate_loaded_trajectory,
    validate_unloaded_trajectory,
)

FAKE_PLUGIN = 'uf_robot_hardware/UFRobotFakeSystemHardware'
ARM_NAMES = tuple(f'joint{i}' for i in range(1, 7))


def _string_parameter(node, remote, name):
    request = GetParameters.Request()
    request.names = [name]
    result = _call_service(node, GetParameters, f'{remote}/get_parameters', request)
    if len(result.values) != 1 or result.values[0].type != ParameterType.PARAMETER_STRING:
        raise RuntimeError(f'Missing string parameter: {remote}/{name}')
    return result.values[0].string_value


def require_fake_environment(node):
    if node.get_parameter('gripper_backend').value != 'fake':
        raise RuntimeError('pickup_fake requires gripper_backend:=fake.')

    require_hardware_environment(node, expected_backend='fake')
    limits = read_arm_joint_limits(node)

    mode = _string_parameter(
        node,
        '/lite6_traj_controller',
        'interpolation_method',
    )

    if mode != 'splines':
        raise RuntimeError(
            'pickup_fake requires controller interpolation_method=splines.'
        )

    return limits

def scene_snapshot(node):
    request = GetPlanningScene.Request()
    # Robot state, attachments, world geometry, octomap and ACM.
    request.components.components = 2 | 4 | 16 | 32 | 128
    return deepcopy(_call_service(
        node, GetPlanningScene, '/get_planning_scene', request,
    ).scene)


def _attachment_map(scene):
    items = scene.robot_state.attached_collision_objects
    if len({a.object.id for a in items}) != len(items):
        raise RuntimeError('Duplicate attached object IDs.')
    return {a.object.id: a for a in items}


def assert_same_scene(actual, expected):
    a = {obj.id: obj for obj in actual.world.collision_objects}
    b = {obj.id: obj for obj in expected.world.collision_objects}
    if (len(a) != len(actual.world.collision_objects)
            or len(b) != len(expected.world.collision_objects)
            or set(a) != set(b)
            or any(not _same_object_geometry(a[name], b[name]) for name in b)):
        raise RuntimeError('World geometry changed since validation. Sequence blocked.')
    aa, bb = _attachment_map(actual), _attachment_map(expected)
    if set(aa) != set(bb) or any(
        aa[name].link_name != bb[name].link_name
        or list(aa[name].touch_links) != list(bb[name].touch_links)
        or not _same_object_geometry(aa[name].object, bb[name].object)
        for name in bb
    ):
        raise RuntimeError('Live attachments changed since validation. Sequence blocked.')
    if (actual.allowed_collision_matrix != expected.allowed_collision_matrix
            or actual.world.octomap != expected.world.octomap):
        raise RuntimeError('Collision matrix or octomap changed. Sequence blocked.')


def _check_world_status(node, planned):
    current = check_planning_ready(node)
    if any(current.get(key) != planned.get(key) for key in ('check_id', 'state')):
        raise RuntimeError('World status changed. Create a new pickup plan.')


def _check_joint_match(state, result, *, endpoint=False, closed=None):
    joint_state = state.joint_state
    if len(joint_state.name) != len(joint_state.position):
        raise RuntimeError('Incomplete measured robot state.')
    positions = dict(zip(joint_state.name, joint_state.position))
    trajectory = result.trajectory.joint_trajectory
    point = trajectory.points[-1 if endpoint else 0]
    expected = dict(zip(trajectory.joint_names, point.positions))
    if closed is not None:
        for name in ('finger_joint1', 'finger_joint2'):
            expected[name] = 0.0 if closed else 0.0089
    for name, value in expected.items():
        actual = positions.get(name)
        if actual is None or not math.isfinite(actual) or abs(actual-value) > 0.0001:
            raise RuntimeError(f'Fake measured state does not match the plan: {name}')


def _wait_measured_state(node, result, *, endpoint=False, closed=None):
    deadline = time.monotonic() + 3.0
    while True:
        state = scene_snapshot(node).robot_state
        try:
            _check_joint_match(state, result, endpoint=endpoint, closed=closed)
            return state
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            rclpy.spin_once(node, timeout_sec=0.05)


def validate_unloaded_segment(node, result, *, label):
    limits = require_fake_environment(node)

    validate_unloaded_trajectory(
        node,
        result,
        limits,
        label=label,
    )

def _attach_live_plate(node, measured_grasp, expected_scene):
    assert_same_scene(scene_snapshot(node), expected_scene)
    request = ApplyPlanningScene.Request()
    request.scene.is_diff = True
    request.scene.robot_state.is_diff = True
    request.scene.robot_state.attached_collision_objects = deepcopy(
        measured_grasp.attached_collision_objects,
    )
    if not _call_service(
        node, ApplyPlanningScene, '/apply_planning_scene', request,
    ).success:
        raise RuntimeError('Live fake plate attachment was rejected.')
    expected = deepcopy(expected_scene)
    expected.world.collision_objects = [
        obj for obj in expected.world.collision_objects if obj.id not in PLATE_IDS
    ]
    expected.robot_state.attached_collision_objects = deepcopy(
        measured_grasp.attached_collision_objects,
    )
    assert_same_scene(scene_snapshot(node), expected)
    node.get_logger().info('LIVE FAKE ATTACH VERIFIED: two attached parts; no world duplicates.')
    return expected


def review_and_run_fake_pickup(node, segments, planned_world_status, original_scene):
    limits = require_fake_environment(node)
    if [label for label, _ in segments] != ['APPROACH', 'DESCENT', 'LOADED LIFT', 'LOADED CARRY']:
        raise RuntimeError('Incomplete fake pickup sequence.')
    for _, result in segments:
        check_trajectory_limits(result, limits)
    # Verify arm continuity, including the open-to-closed grasp transition.
    for (_, first), (_, second) in zip(segments, segments[1:]):
        endpoint = state_at_trajectory_point(first.trajectory_start, first.trajectory)
        _check_joint_match(endpoint, second)
    assert_same_scene(scene_snapshot(node), original_scene)
    publishers = []
    try:
        for label, result in segments:
            node.get_logger().info(f'REVIEW {label}: exact validated path; robot remains stationary.')
            # Only one latched publisher at a time prevents old previews replaying.
            if publishers:
                node.destroy_publisher(publishers.pop())
            publishers.append(show_trajectory(node, result))
            answer = input(f'Inspect the entire {label} in RViz. Enter = next; q = cancel: ').strip()
            if answer:
                node.get_logger().info('Fake sequence cancelled before execution.')
                return
        if not node.get_parameter('execute').value:
            node.get_logger().info('FAKE SEQUENCE PREVIEW COMPLETE. No motion or live attachment.')
            return
        answer = input(
            'All four paths reviewed. Fake gripper closes after descent; '
            'plate stays attached at exit.\n'
            'Type RUN_FAKE to execute this exact sequence; anything else cancels: '
        ).strip()
        if answer != 'RUN_FAKE':
            node.get_logger().info('Fake sequence cancelled before execution.')
            return

        expected_scene = original_scene
        phase = 'before approach'
        try:
            def ready(result, *, closed):
                require_fake_environment(node)
                _check_world_status(node, planned_world_status)
                current = scene_snapshot(node)
                assert_same_scene(current, expected_scene)
                check_execution_start(node, result)
                _check_joint_match(current.robot_state, result, closed=closed)

            for label, result in segments[:2]:
                phase = label
                ready(result, closed=False)
                validate_unloaded_segment(node, result, label=label)
                # Recheck after the validation service calls.
                ready(result, closed=False)
                execute_reviewed_trajectory(
                    node, result, planned_world_status, fake_only=True, confirm=False,
                )
                _wait_measured_state(node, result, endpoint=True, closed=False)

            phase = 'fake gripper close / live attach'
            unloaded_grasp_result = SimpleNamespace(
                trajectory=segments[2][1].trajectory,
                trajectory_start=state_at_trajectory_point(
                    segments[1][1].trajectory_start, segments[1][1].trajectory,
                ),
            )
            ready(unloaded_grasp_result, closed=False)
            close_gripper(node, fake_position=0.0)
            measured = _wait_measured_state(node, unloaded_grasp_result, closed=True)
            require_fake_environment(node)
            _check_world_status(node, planned_world_status)
            assert_same_scene(scene_snapshot(node), expected_scene)
            measured_grasp = build_hypothetical_grasp(node, measured)
            expected_scene = _attach_live_plate(node, measured_grasp, expected_scene)
            loaded_lift = SimpleNamespace(
                trajectory=segments[2][1].trajectory,
                trajectory_start=measured_grasp,
            )
            loaded_carry = SimpleNamespace(
                trajectory=segments[3][1].trajectory,
                trajectory_start=state_at_trajectory_point(
                    measured_grasp, loaded_lift.trajectory,
                ),
            )
            phase = 'loaded paths revalidation with measured attachment geometry'
            # Same arm paths; validate both again against the actual fake grasp.
            for label, result, allow_support in (
                ('LOADED LIFT', loaded_lift, True),
                ('LOADED CARRY', loaded_carry, False),
            ):
                validate_loaded_trajectory(
                    node,
                    SimpleNamespace(start_state=result.trajectory_start,
                                    solution=result.trajectory),
                    result.trajectory_start, label=f'LIVE FAKE {label}',
                    allow_initial_support=allow_support,
                )
            for label, result in (('LOADED LIFT', loaded_lift), ('LOADED CARRY', loaded_carry)):
                phase = label
                ready(result, closed=True)
                execute_reviewed_trajectory(
                    node, result, planned_world_status, fake_only=True, confirm=False,
                )
                _wait_measured_state(node, result, endpoint=True, closed=True)
                assert_same_scene(scene_snapshot(node), expected_scene)
            node.get_logger().info(
                'FAKE PICKUP COMPLETE: approach, descent, close, attach, lift and '
                'carry succeeded. Plate remains attached at the exit. '
                'No automatic opening, detaching or return-home.'
            )
        except (Exception, KeyboardInterrupt):
            node.get_logger().error(
                f'FAKE SEQUENCE STOPPED during {phase}. No automatic reset, '
                'opening or detaching. Inspect the current fake scene before restarting.'
            )
            raise
    finally:
        for publisher in publishers:
            node.destroy_publisher(publisher)
