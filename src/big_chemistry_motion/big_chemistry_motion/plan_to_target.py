import rclpy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetMotionPlan
from shape_msgs.msg import SolidPrimitive

from big_chemistry_motion.target_reader import TargetReader

import time

from moveit_msgs.msg import DisplayTrajectory
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


def plan_to_target(node, transform):
    client = node.create_client(
        GetMotionPlan,
        '/plan_kinematic_path',
    )

    try:
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('MoveIt planning service is unavailable.')

        reference_frame = transform.header.frame_id
        goal_link = 'link_eef'

        # Allow the link origin within a sphere of radius 5 mm.
        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [0.005]

        region_pose = Pose()
        region_pose.position.x = transform.transform.translation.x
        region_pose.position.y = transform.transform.translation.y
        region_pose.position.z = transform.transform.translation.z
        region_pose.orientation.w = 1.0

        position = PositionConstraint()
        position.header.frame_id = reference_frame
        position.link_name = goal_link
        position.constraint_region.primitives = [region]
        position.constraint_region.primitive_poses = [region_pose]
        position.weight = 1.0

        # Orientation tolerance: 0.05 rad, approximately 2.9 degrees.
        orientation = OrientationConstraint()
        orientation.header.frame_id = reference_frame
        orientation.link_name = goal_link
        orientation.orientation = transform.transform.rotation
        orientation.absolute_x_axis_tolerance = 0.05
        orientation.absolute_y_axis_tolerance = 0.05
        orientation.absolute_z_axis_tolerance = 0.05
        orientation.weight = 1.0

        goal = Constraints()
        goal.name = 'device_relative_target'
        goal.position_constraints = [position]
        goal.orientation_constraints = [orientation]

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = 'lite6'
        motion_request.start_state.is_diff = True
        motion_request.goal_constraints = [goal]
        motion_request.num_planning_attempts = 5
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = 0.1
        motion_request.max_acceleration_scaling_factor = 0.1

        node.get_logger().info(
            f'Requesting plan for lite6 / {goal_link}. No execution.'
        )

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)

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

def show_trajectory(node, result):
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    publisher = node.create_publisher(
        DisplayTrajectory,
        '/display_planned_path',
        qos,
    )

    node.get_logger().info('Waiting for a trajectory display subscriber...')

    deadline = time.monotonic() + 10.0

    while rclpy.ok() and publisher.get_subscription_count() == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'No trajectory display subscriber found. '
                'Check the RViz Planned Path display.'
            )

        rclpy.spin_once(node, timeout_sec=0.1)

    if not rclpy.ok():
        return

    display = DisplayTrajectory()
    display.trajectory_start = result.trajectory_start
    display.trajectory = [result.trajectory]

    publisher.publish(display)

    node.get_logger().info(
        'Trajectory preview published. No execution. '
        'Press Ctrl+C to close this node.'
    )

    rclpy.spin(node)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = TargetReader()
        target = node.read_target()
        result = plan_to_target(node, target)
        show_trajectory(node, result)
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