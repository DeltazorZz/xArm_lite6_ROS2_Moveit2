import rclpy
import math

from action_msgs.srv import CancelGoal
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from xarm_msgs.srv import Call


_REAL_SERVICES = {
    'open': '/ufactory/open_lite6_gripper',
    'close': '/ufactory/close_lite6_gripper',
    'stop': '/ufactory/stop_lite6_gripper',
}


def _call_gripper_service(node, service_name):
    client = node.create_client(Call, service_name)
    
    try: 
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(
                f'Gripper service unavailable: {service_name}'
            )

        future = client.call_async(Call.Request())
        rclpy.spin_until_future_complete(
            node,
            future,
            timeout_sec=5.0,
        )

        if not future.done():
            future.cancel()
            raise RuntimeError(
                f'Gripper response timed out: {service_name}. '
                'The command may have been applied. '
                'Do not retry automatically.'
            )

        if future.cancelled():
            raise RuntimeError(
                f'Gripper request cancelled: {service_name}. '
                'Gripper state is unknown.'
            )

        error = future.exception()
        if error is not None:
            raise RuntimeError(
                f'Gripper request failed: {service_name}. '
                'Gripper state is unknown.'
            ) from error

        response = future.result()
        if response is None:
            raise RuntimeError(
                f'No gripper response: {service_name}. '
                'Gripper state is unknown.'
            )

        if response.ret != 0:
            raise RuntimeError(
                f'Gripper command failed: {service_name}; '
                f'ret={response.ret}; message={response.message}'
            )

        node.get_logger().info(
            f'Gripper command returned success: {service_name}. '
            'Physical completion and plate grasp are not confirmed.'
        )

        return response

    finally:
        node.destroy_client(client)
        
def _move_fake_gripper(node, position):
    if (
        isinstance(position, bool)
        or not isinstance(position, (int, float))
        or not math.isfinite(position)
        or not 0.0 <= position <= 0.0089
    ):
        raise ValueError(
            'Fake gripper position must be a number within 0–0.0089 m.'
        )

    position = float(position)

    client = ActionClient(
        node,
        GripperCommand,
        '/lite6_fake_gripper_controller/gripper_cmd',
    )
    goal_handle = None
    result_future = None

    try:
        if not client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError('Fake gripper action server unavailable.')

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 0.0

        goal_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            node, goal_future, timeout_sec=5.0,
        )

        if not goal_future.done():
            raise RuntimeError(
                'Fake gripper acknowledgement timed out. '
                'Command state unknown; do not retry automatically.'
            )

        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('Fake gripper goal was rejected.')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            node, result_future, timeout_sec=5.0,
        )

        if not result_future.done():
            raise RuntimeError('Fake gripper movement timed out.')

        response = result_future.result()
        if response is None:
            raise RuntimeError('No fake gripper result received.')

        result = response.result

        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or not result.reached_goal
            or result.stalled
            or not math.isfinite(result.position)
            or abs(result.position - position) > 0.0001
        ):
            raise RuntimeError(
                f'Fake gripper did not reach its target: '
                f'status={response.status}, '
                f'position={result.position}, '
                f'reached_goal={result.reached_goal}, '
                f'stalled={result.stalled}'
            )

        node.get_logger().info(
            f'Fake gripper reached joint position {result.position:.4f} m.'
        )
        return result

    except (Exception, KeyboardInterrupt):
        if (
            goal_handle is not None
            and goal_handle.accepted
            and (result_future is None or not result_future.done())
            and rclpy.ok()
        ):
            try:
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(
                    node, cancel_future, timeout_sec=3.0,
                )
                node.get_logger().warning(
                    'Fake gripper cancellation requested; '
                    'stopping is not confirmed.'
                )
            except Exception as error:
                node.get_logger().error(
                    f'Fake gripper cancellation failed: {error}'
                )
        raise

    finally:
        client.destroy()
        
def _command_gripper(node, command, fake_position=None):
    if command not in _REAL_SERVICES:
        raise ValueError(f'Unknown gripper command: {command}')

    if not node.has_parameter('gripper_backend'):
        raise RuntimeError(
            'gripper_backend parameter is missing. '
            'Select real or fake explicitly.'
        )

    backend = node.get_parameter('gripper_backend').value

    if backend not in ('real', 'fake'):
        raise ValueError(
            'gripper_backend must be "real" or "fake".'
        )

    if backend == 'real':
        if fake_position is not None:
            raise ValueError(
                'fake_position cannot be used with the real backend.'
            )

        return _call_gripper_service(node, _REAL_SERVICES[command])

    if command == 'open':
        return _move_fake_gripper(node, 0.0089)

    if command == 'close':
        if fake_position is None:
            raise ValueError(
                'Fake closing requires an explicit fake_position.'
            )

        return _move_fake_gripper(node, fake_position)

    
    return _stop_fake_gripper(node)
    


def open_gripper(node):
    return _command_gripper(node, 'open')

def close_gripper(node, fake_position=None):
    return _command_gripper(
        node,
        'close',
        fake_position=fake_position,
    )
    
def stop_gripper(node):
    return _command_gripper(node, 'stop')
    
def _stop_fake_gripper(node):
    client = node.create_client(
        CancelGoal,
        '/lite6_fake_gripper_controller/gripper_cmd/_action/cancel_goal',
    )

    try:
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('Fake gripper cancel service unavailable.')

        # Zero goal ID and timestamp: cancel all goals on this action server.
        future = client.call_async(CancelGoal.Request())
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=3.0,
        )

        if not future.done():
            future.cancel()
            raise RuntimeError(
                'Fake gripper cancellation timed out. '
                'Stopping is not confirmed.'
            )

        if future.cancelled():
            raise RuntimeError('Fake gripper cancel request was cancelled.')

        error = future.exception()
        if error is not None:
            raise RuntimeError(
                'Fake gripper cancel request failed.'
            ) from error

        response = future.result()
        if response is None:
            raise RuntimeError('No fake gripper cancellation response.')

        if response.return_code != CancelGoal.Response.ERROR_NONE:
            raise RuntimeError(
                'Fake gripper cancellation was not accepted: '
                f'code={response.return_code}. '
                'Stopping is not confirmed.'
            )

        count = len(response.goals_canceling)

        if count:
            node.get_logger().warning(
                f'Cancellation accepted for {count} fake gripper goal(s). '
                'Stopping is not confirmed yet.'
            )
        else:
            node.get_logger().info(
                'No fake gripper goals entered cancellation. '
                'This does not confirm the gripper state.'
            )

        return response

    finally:
        node.destroy_client(client)
        
def check_real_gripper_services(node):
    for command, service_name in _REAL_SERVICES.items():
        client = node.create_client(Call, service_name)

        try:
            if not client.wait_for_service(timeout_sec=3.0):
                raise RuntimeError(
                    f'Real gripper service unavailable: {service_name}'
                )

            node.get_logger().info(
                f'Real gripper service available: {command}'
            )
        finally:
            node.destroy_client(client)