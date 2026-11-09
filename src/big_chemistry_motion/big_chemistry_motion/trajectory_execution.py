import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.action import ActionClient

from big_chemistry_motion.motion_checks import (
    check_planning_ready,
    check_execution_start,
)
from big_chemistry_motion.motion_checks import (
    check_planning_ready,
    check_execution_start,
    check_controller_ready,
)


def execute_reviewed_trajectory(node, result, planned_world_status):
    client = ActionClient(
        node,
        ExecuteTrajectory,
        '/execute_trajectory',
    )
    goal_handle = None
    result_future = None
    request_sent = False

    try:
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError('Execution action server is unavailable.')

        answer = input(
            'Review the complete trajectory in RViz.\n'
            'Type EXECUTE to move the real robot; anything else cancels: '
        ).strip()

        if answer != 'EXECUTE':
            node.get_logger().info('Cancelled. No motion command sent.')
            return

        current_status = check_planning_ready(node)

        if (
            current_status.get('check_id')
            != planned_world_status.get('check_id')
            or current_status.get('state')
            != planned_world_status.get('state')
        ):
            raise RuntimeError(
                'World status changed. Create and review a new plan.'
            )

        check_controller_ready(node)
        check_execution_start(node, result)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.trajectory

        node.get_logger().warning(
            'Sending reviewed trajectory. The real robot may move now.'
        )

        goal_future = client.send_goal_async(goal)
        request_sent = True
        rclpy.spin_until_future_complete(
            node, goal_future, timeout_sec=10.0,
        )

        if not goal_future.done():
            raise RuntimeError(
                'Execution acknowledgement timed out. '
                'Motion state UNKNOWN; use the physical stop. Do not retry.'
            )

        goal_handle = goal_future.result()

        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('MoveIt rejected the execution request.')

        result_future = goal_handle.get_result_async()

        last_time = result.trajectory.joint_trajectory.points[-1].time_from_start
        duration = last_time.sec + last_time.nanosec * 1e-9

        rclpy.spin_until_future_complete(
            node,
            result_future,
            timeout_sec=max(30.0, duration * 2.0 + 10.0),
        )

        if not result_future.done():
            raise RuntimeError('Execution result timed out.')

        response = result_future.result()
        if response is None:
            raise RuntimeError('No execution result received.')

        code = response.result.error_code.val

        if (
            response.status != GoalStatus.STATUS_SUCCEEDED
            or code != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError(
                f'Execution failed: action status={response.status}, '
                f'MoveIt error={code}.'
            )

        node.get_logger().info('Execution completed successfully.')

    except (Exception, KeyboardInterrupt):
        if request_sent:
            node.get_logger().error(
                'Execution interrupted or failed. '
                'Verify the robot has stopped; use the physical stop if needed.'
            )

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
                    'Cancellation requested. This is not confirmation '
                    'that the robot has physically stopped.'
                )
            except Exception as cancel_error:
                node.get_logger().error(
                    f'Could not request cancellation: {cancel_error}'
                )
        raise

    finally:
        client.destroy()