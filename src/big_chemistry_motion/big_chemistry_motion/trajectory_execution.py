import rclpy
from action_msgs.msg import GoalStatus
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from rclpy.action import ActionClient

from big_chemistry_motion.motion_checks import (
    check_planning_ready,
    check_execution_start,
    check_controller_ready,
)

def _require_execution_backend(node, expected_backend):
    if expected_backend not in ('fake', 'real'):
        raise ValueError('Execution backend must be fake or real.')

    selected = node.get_parameter('gripper_backend').value

    if selected != expected_backend:
        raise RuntimeError(
            f'Execution expects backend={expected_backend}, '
            f'but gripper_backend={selected}.'
        )

    if expected_backend == 'fake':
        from big_chemistry_motion.fake_pickup import (
            require_fake_environment,
        )
        require_fake_environment(node)

    else:
        from big_chemistry_motion.real_pickup import (
            require_real_environment,
        )
        require_real_environment(node)

def execute_reviewed_trajectory(
    node,
    result,
    planned_world_status,
    *,
    fake_only=False,
    confirm=True,
    expected_backend=None,
):
    if fake_only and expected_backend not in (None, 'fake'):
        raise ValueError(
            'fake_only conflicts with the requested execution backend.'
        )

    if fake_only:
        backend = 'fake'
    elif expected_backend is None:
        backend = node.get_parameter('gripper_backend').value
    else:
        backend = expected_backend

    _require_execution_backend(node, backend)

    if not node.get_parameter('execute').value:
        raise RuntimeError('Trajectory execution requires execute:=true.')

    if fake_only:
        if node.get_parameter('motion_goal').value != 'pickup_fake':
            raise RuntimeError(
                'Fake sequence execution requires motion_goal:=pickup_fake.'
            )

    elif not confirm:
        raise RuntimeError(
            'Skipping individual confirmation is restricted '
            'to the reviewed fake sequence.'
        )

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

        if confirm:
            robot_label = 'FAKE robot' if backend == 'fake' else 'REAL robot'

            answer = input(
                'Review the complete trajectory in RViz.\n'
                f'Type EXECUTE to move the {robot_label}; '
                'anything else cancels: '
            ).strip()

            if answer != 'EXECUTE':
                node.get_logger().info('Cancelled. No motion command sent.')
                return False

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
        _require_execution_backend(node, backend)
        check_execution_start(node, result)
        
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = result.trajectory


        node.get_logger().warning(
            f'Sending reviewed trajectory to the verified {backend} backend. \n'
            'The real robot may move now.'
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
        return True

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
