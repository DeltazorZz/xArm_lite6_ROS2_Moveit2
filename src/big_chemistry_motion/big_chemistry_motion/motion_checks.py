import json
import math
import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from copy import deepcopy
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import AllowedCollisionEntry
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene



def check_planning_ready(node):
    client = node.create_client(
        Trigger,
        '/big_chemistry_world/position_check_status',
    )
    
    try: 
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                'World status service is unavailable. Planning blocked.'
            )
        
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(
            node,
            future,
            timeout_sec=5.0,
        )
        
        if not future.done():
            future.cancel()
            raise RuntimeError(
                'Timed out waiting for world status. Planning blocked.'
            )

        if future.cancelled():
            raise RuntimeError('World status request was cancelled.')
        
        if future.exception() is not None:
            raise RuntimeError(
                  f'World status request failed: {future.exception()}'
            )
            
        response = future.result()
        
        if response is None or not response.success:
            raise RuntimeError(
                 'Could not query world status. Planning blocked.'
            )    
            
        try:
            status = json.loads(response.message)    
        except json.JSONDecodeError as error: 
            raise RuntimeError(
                'World status response is not valid JSON.'
            )from error
            
        if not isinstance (status, dict):
            raise RuntimeError(
                'World status response must be a JSON object.'
            )
            
        if (
            status.get('scene_confirmed') is not True
            or status.get('planning_ready') is not True
        ):
            raise RuntimeError(
                f'Planning blocked: state={status.get("state")}, '
                f'detail={status.get("detail")}'
            )
            
        node.get_logger().info(
            f'World ready for planning. '
            f'Position check #{status.get("check_id")}: '
            f'{status.get("state")}.'
        ) 
        
        return status
    
    finally:
        node.destroy_client(client)
        
def check_controller_ready(node):
    client = node.create_client(
        ListControllers,
        '/controller_manager/list_controllers',
    )

    try:
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(
                'Controller manager unavailable. Execution blocked.'
            )

        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=3.0,
        )

        if not future.done():
            future.cancel()
            raise RuntimeError(
                'Controller status request timed out. Execution blocked.'
            )

        if future.cancelled():
            raise RuntimeError('Controller status request was cancelled.')

        if future.exception() is not None:
            raise RuntimeError(
                f'Controller status request failed: {future.exception()}'
            )

        response = future.result()
        if response is None:
            raise RuntimeError('No controller status received.')

        controller = next(
            (
                item for item in response.controller
                if item.name == 'lite6_traj_controller'
            ),
            None,
        )

        if controller is None:
            raise RuntimeError(
                'lite6_traj_controller is not loaded. Execution blocked.'
            )

        if controller.state != 'active':
            raise RuntimeError(
                f'lite6_traj_controller is "{controller.state}". '
                'Execution blocked.'
            )

        node.get_logger().info('Trajectory controller is active.')

    finally:
        node.destroy_client(client)
        
def check_execution_interfaces(node):
    from rclpy.action import ActionClient
    from moveit_msgs.action import ExecuteTrajectory
    from control_msgs.action import FollowJointTrajectory

    interfaces = (
        (ExecuteTrajectory, '/execute_trajectory'),
        (
            FollowJointTrajectory,
            '/lite6_traj_controller/follow_joint_trajectory',
        ),
    )

    for action_type, action_name in interfaces:
        client = ActionClient(node, action_type, action_name)

        try:
            if not client.wait_for_server(timeout_sec=3.0):
                raise RuntimeError(
                    f'Execution action unavailable: {action_name}'
                )

            node.get_logger().info(
                f'Execution action available: {action_name}'
            )
        finally:
            client.destroy()    
        
def check_execution_start(node, result):
    trajectory = result.trajectory.joint_trajectory
    expected = dict(zip(
        trajectory.joint_names,
        trajectory.points[0].positions,
    ))

    # A gripper állapotát is összevetjük a tervezéskorival.
    start = result.trajectory_start.joint_state
    planned_state = dict(zip(start.name, start.position))

    if 'finger_joint1' not in planned_state:
        raise RuntimeError('Planned start state has no gripper position.')

    expected['finger_joint1'] = planned_state['finger_joint1']
    received = []

    subscription = node.create_subscription(
        JointState,
        '/joint_states',
        lambda message: received.append(message),
        qos_profile_sensor_data,
    )

    try:
        deadline = time.monotonic() + 3.0

        while rclpy.ok() and not received:
            if time.monotonic() >= deadline:
                raise RuntimeError('No fresh joint state received.')
            rclpy.spin_once(node, timeout_sec=0.1)

        if not received:
            raise RuntimeError('Joint state check interrupted.')

        message = received[-1]

        if len(message.name) != len(message.position):
            raise RuntimeError('Incomplete joint state message.')

        current = dict(zip(message.name, message.position))

        for name, planned in expected.items():
            actual = current.get(name)

            if actual is None or not math.isfinite(actual):
                raise RuntimeError(f'Missing or invalid joint state: {name}')

            if not math.isfinite(planned):
                raise RuntimeError(f'Invalid planned joint position: {name}')

            # Kar: 0.01 rad ≈ 0.57°. Gripper: 0.5 mm.
            tolerance = 0.0005 if name == 'finger_joint1' else 0.01

            if abs(actual - planned) > tolerance:
                raise RuntimeError(
                    f'Start state changed: {name}. '
                    'Create and review a new plan.'
                )

        node.get_logger().info('Start state matches the reviewed trajectory.')

    finally:
        node.destroy_subscription(subscription)


def check_initial_support_contact(node, validity_client, validity_request):
    """Temporarily allow only plate_base <-> pickup_slot."""
    get_client = node.create_client(GetPlanningScene, '/get_planning_scene')
    apply_client = node.create_client(ApplyPlanningScene, '/apply_planning_scene')

    def call(client, request):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Support-contact service unavailable.')

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

        if not future.done():
            future.cancel()
            raise RuntimeError('Support-contact service timed out.')

        response = future.result()
        if response is None:
            raise RuntimeError('No support-contact service response.')
        return response

    def apply_matrix(matrix):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.allowed_collision_matrix = matrix
        if not call(apply_client, request).success:
            raise RuntimeError('Could not update collision matrix.')

    try:
        request = GetPlanningScene.Request()
        request.components.components = 128
        original = deepcopy(
            call(get_client, request).scene.allowed_collision_matrix
        )
        modified = deepcopy(original)

        # Avoid changing the meaning of unexpanded default rules.
        if modified.default_entry_names:
            raise RuntimeError(
                'ACM has default rules; inspect before adding support contact.'
            )

        for name in ('plate_base', 'pickup_slot'):
            if name not in modified.entry_names:
                for row in modified.entry_values:
                    row.enabled = list(row.enabled) + [False]

                modified.entry_names.append(name)
                modified.entry_values.append(
                    AllowedCollisionEntry(
                        enabled=[False] * len(modified.entry_names)
                    )
                )

        i = modified.entry_names.index('plate_base')
        j = modified.entry_names.index('pickup_slot')
        modified.entry_values[i].enabled[j] = True
        modified.entry_values[j].enabled[i] = True

        try:
            apply_matrix(modified)
            return call(validity_client, validity_request)
        finally:
            # Restore even when the validity check fails.
            apply_matrix(original)

    finally:
        node.destroy_client(get_client)
        node.destroy_client(apply_client)
        
def is_tiny_support_contact(result):
    """Recognise only the measured initial plate/support contact."""
    return (
        not result.valid
        and bool(result.contacts)
        and all(
            {contact.contact_body_1, contact.contact_body_2}
            == {'plate_base', 'pickup_slot'}
            and math.isfinite(contact.depth)
            and 0.0 <= contact.depth <= 0.000001
            for contact in result.contacts
        )
    )


def diagnose_cartesian_collisions(node, response, *, allow_initial_support=None):
    if allow_initial_support is None:
        allow_initial_support = (
            node.get_parameter('cartesian_phase').value == 'lift'
        )
    client = node.create_client(
        GetStateValidity,
        '/check_state_validity',
    )

    try:
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('State validity service unavailable.')

        trajectory = response.solution.joint_trajectory

        start_positions = dict(zip(
            response.start_state.joint_state.name,
            response.start_state.joint_state.position,
        ))

        for index, point in enumerate(trajectory.points):
            state = deepcopy(response.start_state)
            positions = start_positions.copy()

            positions.update(zip(
                trajectory.joint_names,
                point.positions,
            ))

            state.joint_state.name = list(positions.keys())
            state.joint_state.position = list(positions.values())
            state.joint_state.velocity = []
            state.joint_state.effort = []

            request = GetStateValidity.Request()
            request.robot_state = state
            request.group_name = 'lite6'

            future = client.call_async(request)
            rclpy.spin_until_future_complete(
                node, future, timeout_sec=5.0
            )

            if not future.done():
                future.cancel()
                raise RuntimeError('State validity check timed out.')

            result = future.result()
            if result is None:
                raise RuntimeError('No state validity response.')

            # Only retry a tiny support contact at the initial lift state.
            if (
                index == 0
                and allow_initial_support
                and {'plate_base', 'plate_body'}.issubset(
                    item.object.id for item in state.attached_collision_objects
                )
                and is_tiny_support_contact(result)
            ):
                result = check_initial_support_contact(node, client, request)

                if result.valid:
                    node.get_logger().info(
                        'Initial support contact accepted; original ACM restored.'
                    )


            if not result.valid:
                node.get_logger().warning(
                    f'INVALID state at trajectory point '
                    f'{index + 1}/{len(trajectory.points)}'
                )

                for contact in result.contacts:
                    node.get_logger().warning(
                        f'COLLISION: {contact.contact_body_1} '
                        f'<-> {contact.contact_body_2}, '
                        f'depth={contact.depth:.5f} m'
                    )

                return False

        node.get_logger().info(
            'All sampled trajectory states are collision-free.'
        )
        return True

    finally:
        node.destroy_client(client)


def require_gripper_open(node, min_open_position=0.0084):
    """Verify the simulated gripper is open before pickup descent."""

    received = []

    def callback(message):
        if 'finger_joint1' not in message.name:
            return

        index = message.name.index('finger_joint1')

        if index < len(message.position):
            received.append(message.position[index])

    subscription = node.create_subscription(
        JointState,
        '/joint_states',
        callback,
        qos_profile_sensor_data,
    )

    try:
        deadline = time.monotonic() + 3.0

        while not received and rclpy.ok():
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            rclpy.spin_once(
                node,
                timeout_sec=min(0.2, remaining),
            )

        if not received:
            raise RuntimeError(
                'Gripper state unavailable. Pickup rejected.'
            )

        position = received[-1]

        if (
            not math.isfinite(position)
            or not min_open_position <= position <= 0.0090
        ):
            raise RuntimeError(
                f'Gripper is not fully open: {position:.5f} m. '
                'Pickup rejected.'
            )

        node.get_logger().info(
            f'Gripper OPEN confirmed: {position:.5f} m.'
        )

    finally:
        node.destroy_subscription(subscription)


def check_arm_preview_start(node, result):
    """Compare the preview's arm start with newly received joint data."""
    trajectory = result.trajectory.joint_trajectory
    arm_names = {f'joint{i}' for i in range(1, 7)}

    if (
        len(trajectory.joint_names) != 6
        or set(trajectory.joint_names) != arm_names
        or not trajectory.points
    ):
        raise RuntimeError(
            'Arm preview must contain exactly the six arm joints.'
        )

    first = trajectory.points[0]

    if len(first.positions) != 6:
        raise RuntimeError('Arm preview start positions are incomplete.')

    expected = dict(zip(trajectory.joint_names, first.positions))

    if not all(math.isfinite(value) for value in expected.values()):
        raise RuntimeError('Arm preview start contains invalid positions.')

    received = []

    def callback(message):
        if arm_names.issubset(message.name):
            received.append(message)

    subscription = node.create_subscription(
        JointState,
        '/joint_states',
        callback,
        qos_profile_sensor_data,
    )

    try:
        deadline = time.monotonic() + 3.0

        while not received and rclpy.ok():
            remaining = deadline - time.monotonic()

            if remaining <= 0.0:
                break

            rclpy.spin_once(node, timeout_sec=min(0.2, remaining))

        if not received:
            raise RuntimeError('No complete arm joint state received.')

        message = received[-1]

        if (
            len(message.name) != len(message.position)
            or len(set(message.name)) != len(message.name)
        ):
            raise RuntimeError('Arm joint state is malformed.')

        current = dict(zip(message.name, message.position))

        for name, planned in expected.items():
            actual = current[name]

            if not math.isfinite(actual):
                raise RuntimeError(f'Invalid arm joint feedback: {name}')

            if abs(actual - planned) > 0.01:
                raise RuntimeError(
                    f'Arm preview start changed: {name}. '
                    'Create and review a new plan.'
                )

        node.get_logger().info(
            'Arm preview start matches current joint feedback. '
            'Gripper state is checked separately.'
        )

    finally:
        node.destroy_subscription(subscription)