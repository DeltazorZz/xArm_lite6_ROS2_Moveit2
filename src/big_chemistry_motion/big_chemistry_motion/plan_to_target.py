import rclpy
from big_chemistry_motion.target_reader import TargetReader
from big_chemistry_motion.motion_goals import (build_home_goal, build_target_goal)
from big_chemistry_motion.trajectory_tools import (
    log_trajectory_metrics,
    show_trajectory,
)
from big_chemistry_motion.motion_checks import (
    check_planning_ready,
    check_execution_start,
)
from big_chemistry_motion.trajectory_execution import (
    execute_reviewed_trajectory,
)

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan




def plan_to_target(node, transform=None):
    client = node.create_client(
        GetMotionPlan,
        '/plan_kinematic_path',
    )

    try:
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('MoveIt planning service is unavailable.')
        if transform is None:
            goal = build_home_goal()
        else: 
          goal = build_target_goal(transform)

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = 'lite6'
        motion_request.pipeline_id = 'ompl'
        motion_request.planner_id = 'RRTConnect'
        motion_request.start_state.is_diff = True
        motion_request.goal_constraints = [goal]
        motion_request.num_planning_attempts = 10
        motion_request.allowed_planning_time = 30.0
        motion_request.max_velocity_scaling_factor = 0.1
        motion_request.max_acceleration_scaling_factor = 0.1

        node.get_logger().info(
            f'Requesting plan for lite6 / {goal.name}. No execution.'
        )

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=motion_request.allowed_planning_time + 10.0)

        if not future.done():
            future.cancel()
            raise RuntimeError('Timed out waiting for planning result.')

        response = future.result()
        if response is None:
            raise RuntimeError('No planning response received.')

        result = response.motion_plan_response
        error_code = result.error_code.val

        if error_code != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f'MoveIt planning failed. Error code: {error_code}'
            )

        points = result.trajectory.joint_trajectory.points
        if not points:
            raise RuntimeError(
                'MoveIt reported success but returned no joint trajectory points.'
            )

        node.get_logger().info(
            f'Planning succeeded: {len(points)} trajectory points, '
            f'planning time {result.planning_time:.3f} s.'
        )
        node.get_logger().info('Trajectory was NOT executed.')

        return result

    finally:
        node.destroy_client(client)
        


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = TargetReader()
        node.declare_parameter('execute', False)
        node.declare_parameter('motion_goal', 'target')
        node.declare_parameter('gripper_empty_confirmed', False)

        motion_goal = node.get_parameter('motion_goal').value

        if motion_goal not in ('target', 'home'):
            raise ValueError('motion_goal must be "target" or "home".')

        if motion_goal == 'home':
            if not node.get_parameter('gripper_empty_confirmed').value:
                raise ValueError(
                    'Home requires an empty gripper. '
                    'Set gripper_empty_confirmed:=true after checking.'
                )

        planned_world_status = check_planning_ready(node)

        if motion_goal == 'home':
            result = plan_to_target(node)
        else:
            target = node.read_target()
            result = plan_to_target(node, target)
            
        log_trajectory_metrics(node, result)
        preview_publisher = show_trajectory(node, result)

        if node.get_parameter('execute').value:
            execute_reviewed_trajectory(
                node,
                result,
                planned_world_status,
            )
        else:
            check_execution_start(node, result)
            node.get_logger().info(
                'Preview only. Press Ctrl+C to close.'
            )
            rclpy.spin(node)
    except (ValueError, RuntimeError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()