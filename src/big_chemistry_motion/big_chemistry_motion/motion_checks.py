import json
import math
import time

import rclpy
from controller_manager_msgs.srv import ListControllers
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

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