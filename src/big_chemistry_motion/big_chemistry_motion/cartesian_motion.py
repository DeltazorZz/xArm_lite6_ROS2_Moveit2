import rclpy
from copy import deepcopy
import math

from big_chemistry_motion.motion_goals import build_target_pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath

from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from moveit_msgs.srv import GetPlanningScene, GetPositionFK

def plan_cartesian_to_target(node, transform, *, start_state=None):
    client = node.create_client(
        GetCartesianPath,
        '/compute_cartesian_path',
    )
    
    try: 
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('Cartesian planning service is unavailable.')
        
        target = build_target_pose(transform)
        
        request = GetCartesianPath.Request()
        request.header.frame_id = transform.header.frame_id
        
        if start_state is None: 
            request.start_state.is_diff = True
        else: 
            request.start_state = deepcopy(start_state)
        
        request.group_name = 'lite6'
        request.link_name = 'link_eef'
        request.waypoints = [target]
        request.max_step = 0.001
        request.jump_threshold = 0.0
        request.avoid_collisions = True
      
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1
        
        node.get_logger().info(
            'Planning Cartesian path from '
            + (
                'the current robot state.'
                if start_state is None
                else 'a supplied hypothetical robot state. No motion.'
            )
        )
        
        node.get_logger().info(
            f'Cartesian settings: '
            f'avoid_collisions={request.avoid_collisions}, '
            f'max_step={request.max_step}, '
            f'jump_threshold={request.jump_threshold}'
        )

        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=30.0,
            
        )
        
        if not future.done():
            future.cancel()
            raise RuntimeError('Cartesian planning response timed out.')
        
        response = future.result()
        if response is None: 
            raise RuntimeError(
                'No Cartesian planning response received'
            )
            
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f'Cartesian planning failed: {response.error_code.val}'
            )
            
        node.get_logger().info(
            f'Cartesian path completed: {response.fraction:.2%}'
        )
        
        if not 0.999999999 <= response.fraction <= 1.0:
            if start_state is not None:
                _diagnose_incomplete_loaded_precheck(node, request)
            raise RuntimeError(
                'Cartesian path is incomplete. No execution.'
            )
            
        if not response.solution.joint_trajectory.points:
            raise RuntimeError('Cartesian planner returned an empty trajectory.')
        
        return response
    
    finally: 
        node.destroy_client(client)


def _diagnose_incomplete_loaded_precheck(node, original_request):
    if (
        node.get_parameter('motion_goal').value != 'precheck'
        or node.get_parameter('execute').value
        or {item.object.id for item in
            original_request.start_state.attached_collision_objects}
        != {'plate_base', 'plate_body'}
    ):
        return

    # The enclosing precheck scene context remains active throughout this call.
    # Do not replan approach or descent: retain exactly this candidate's state.
    diagnostic_request = deepcopy(original_request)
    diagnostic_request.avoid_collisions = False
    node.get_logger().warning(
        'DIAGNOSTIC ONLY: computing an IK-only path from the SAME loaded '
        'pickup state and target. This path will be collision-checked, '
        'then discarded. It cannot be executed.'
    )
    try:
        response = _call_service(
            node, GetCartesianPath, '/compute_cartesian_path', diagnostic_request,
        )
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            node.get_logger().warning(
                f'DIAGNOSTIC IK-only request failed: {response.error_code.val}'
            )
            return

        points = response.solution.joint_trajectory.points
        node.get_logger().info(
            f'DIAGNOSTIC IK-only fraction: {response.fraction:.2%}; '
            f'{len(points)} timed trajectory points. No execution.'
        )
        if not points:
            node.get_logger().warning('DIAGNOSTIC returned no states to inspect.')
            return

        # Humble may fill response.start_state from its modified IK working
        # state. Reconstruct from our own original state and the first point.
        response.start_state = state_at_trajectory_point(
            original_request.start_state, response.solution, point_index=0,
        )
        from big_chemistry_motion.motion_checks import diagnose_cartesian_collisions

        if diagnose_cartesian_collisions(
            node, response, allow_initial_support=True,
        ):
            node.get_logger().warning(
                'DIAGNOSTIC: no collision found in the returned timed samples. '
                'This does not prove the collision-checked path is feasible; '
                'IK or intermediate-state investigation may still be needed.'
            )
    except Exception as error:
        node.get_logger().warning(f'DIAGNOSTIC could not complete: {error}')
    finally:
        node.get_logger().info(
            'DIAGNOSTIC path discarded. The original incomplete plan remains blocked.'
        )
        
        
def state_at_trajectory_point(reference_state, trajectory, point_index=-1):
    joint_trajectory = trajectory.joint_trajectory

    if not joint_trajectory.points:
        raise RuntimeError('Cannot build a state from an empty trajectory.')

    point = joint_trajectory.points[point_index]

    if len(joint_trajectory.joint_names) != len(point.positions):
        raise RuntimeError('Trajectory joint names and positions do not match.')

    state = deepcopy(reference_state)
    joints = state.joint_state

    if len(joints.name) != len(joints.position):
        raise RuntimeError('Reference robot state is incomplete.')

    positions = dict(zip(joints.name, joints.position))
    positions.update(zip(joint_trajectory.joint_names, point.positions))

    joints.name = list(positions)
    joints.position = list(positions.values())
    joints.velocity = []
    joints.effort = []

    # This is a hypothetical state, not a timestamped sensor measurement.
    joints.header.stamp.sec = 0
    joints.header.stamp.nanosec = 0

    return state

def _call_service(node, service_type, service_name, request):
    client = node.create_client(service_type, service_name)

    try:
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'Service unavailable: {service_name}')

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)

        if not future.done():
            future.cancel()
            raise RuntimeError(f'Service timed out: {service_name}')

        response = future.result()
        if response is None:
            raise RuntimeError(f'Empty response: {service_name}')

        return response
    finally:
        node.destroy_client(client)


def _quaternion_tuple(q):
    values = (q.x, q.y, q.z, q.w)
    norm = math.sqrt(sum(value * value for value in values))

    if not math.isfinite(norm) or norm < 1e-12:
        raise RuntimeError('Invalid quaternion.')

    return tuple(value / norm for value in values)


def _quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def build_hypothetical_grasp(node, pickup_state):
    scene_request = GetPlanningScene.Request()
    scene_request.components.components = 20  # Attachments + world geometry
    scene = _call_service(
        node, GetPlanningScene, '/get_planning_scene', scene_request
    ).scene

    if (
        pickup_state.attached_collision_objects
        or scene.robot_state.attached_collision_objects
    ):
        raise RuntimeError('Grasp precheck requires an unloaded scene.')

    objects = {
        obj.id: obj
        for obj in scene.world.collision_objects
    }
    plate_ids = ('plate_base', 'plate_body')

    for object_id in plate_ids:
        if object_id not in objects:
            raise RuntimeError(f'Missing world object: {object_id}')
        if objects[object_id].header.frame_id != 'world':
            raise RuntimeError(f'{object_id} must be expressed in world.')

    state = deepcopy(pickup_state)
    joints = state.joint_state
    positions = dict(zip(joints.name, joints.position))

    for name in ('finger_joint1', 'finger_joint2'):
        if name not in positions:
            raise RuntimeError(f'Missing gripper joint: {name}')
        positions[name] = 0.0  # Fake closed-gripper model

    joints.name = list(positions)
    joints.position = list(positions.values())
    joints.velocity = []
    joints.effort = []
    state.is_diff = False

    fk_request = GetPositionFK.Request()
    fk_request.header.frame_id = 'world'
    fk_request.fk_link_names = ['link_eef']
    fk_request.robot_state = state

    fk_response = _call_service(
        node, GetPositionFK, '/compute_fk', fk_request
    )

    if (
        fk_response.error_code.val != MoveItErrorCodes.SUCCESS
        or len(fk_response.pose_stamped) != 1
    ):
        raise RuntimeError('FK failed for the planned pickup state.')

    tool = fk_response.pose_stamped[0]
    if tool.header.frame_id != 'world':
        raise RuntimeError('FK returned an unexpected reference frame.')

    q = _quaternion_tuple(tool.pose.orientation)
    inverse_q = (-q[0], -q[1], -q[2], q[3])

    for object_id in plate_ids:
        obj = deepcopy(objects[object_id])

        displacement = (
            obj.pose.position.x - tool.pose.position.x,
            obj.pose.position.y - tool.pose.position.y,
            obj.pose.position.z - tool.pose.position.z,
            0.0,
        )

        local_position = _quaternion_multiply(
            _quaternion_multiply(inverse_q, displacement), q
        )
        local_orientation = _quaternion_multiply(
            inverse_q, _quaternion_tuple(obj.pose.orientation)
        )

        obj.header.frame_id = 'link_eef'
        obj.header.stamp.sec = 0
        obj.header.stamp.nanosec = 0
        obj.operation = CollisionObject.ADD

        (
            obj.pose.position.x,
            obj.pose.position.y,
            obj.pose.position.z,
        ) = local_position[:3]

        (
            obj.pose.orientation.x,
            obj.pose.orientation.y,
            obj.pose.orientation.z,
            obj.pose.orientation.w,
        ) = local_orientation

        attached = AttachedCollisionObject()
        attached.link_name = 'link_eef'
        attached.object = obj
        attached.touch_links = ['uflite_finger1', 'uflite_finger2']
        state.attached_collision_objects.append(attached)

        node.get_logger().info(
            f'Hypothetical attachment: {object_id}, '
            f'link_eef offset [m]: '
            f'{local_position[0]:.6f}, '
            f'{local_position[1]:.6f}, '
            f'{local_position[2]:.6f}'
        )

    return state
