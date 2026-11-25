import rclpy
import math
from moveit_msgs.msg import (
    Constraints, JointConstraint, MoveItErrorCodes, PlanningSceneComponents,
)
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene, GetPositionIK
from copy import deepcopy
from types import SimpleNamespace
from rclpy.parameter import Parameter

from big_chemistry_motion.target_reader import TargetReader

from big_chemistry_motion.motion_goals import (
    build_home_goal,
    build_target_goal,
    build_diagnostic_joint_goal,
    build_target_pose,
)

from big_chemistry_motion.trajectory_tools import (
    log_trajectory_metrics,
    show_trajectory,
)

from big_chemistry_motion.motion_checks import (
    check_planning_ready,
    check_execution_start,
    diagnose_cartesian_collisions,    
    require_gripper_open,
    check_arm_preview_start,
)

from big_chemistry_motion.trajectory_execution import (
    execute_reviewed_trajectory,
)

from big_chemistry_motion.cartesian_motion import (
    _call_service,
    build_hypothetical_grasp,
    plan_cartesian_to_target,
    state_at_trajectory_point,
)
from big_chemistry_motion.precheck_scene import (
    check_loaded_pickup_state,
    temporary_plate_world_removal,
)
from big_chemistry_motion.loaded_validation import (
    validate_loaded_trajectory,
    validate_unloaded_trajectory,
)
from big_chemistry_motion.joint_carry import plan_joint_carry_probe

def plan_to_target(
    node, transform=None, diagnostic=False, *, goal_override=None,
    start_state=None, path_constraints=None, planning_time=60.0,
):
    if not math.isfinite(planning_time) or planning_time <= 0.0:
        raise ValueError('Planning time must be finite and positive.')
    client = node.create_client(
        GetMotionPlan,
        '/plan_kinematic_path',
    )

    try:
        if not client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('MoveIt planning service is unavailable.')
        
        if goal_override is not None:
            goal = deepcopy(goal_override)
        elif diagnostic:
            goal = build_diagnostic_joint_goal()
        elif transform is None:
            goal = build_home_goal()
        else:
            goal = build_target_goal(transform)

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = 'lite6'
        motion_request.pipeline_id = 'ompl'
        motion_request.planner_id = 'RRTConnect'
        if start_state is None:
            motion_request.start_state.is_diff = True
        else:
            motion_request.start_state = deepcopy(start_state)
        if path_constraints is not None:
            motion_request.path_constraints = deepcopy(path_constraints)
        motion_request.goal_constraints = [goal]
        motion_request.num_planning_attempts = 1
        motion_request.allowed_planning_time = float(planning_time)
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


def plan_level_carry(node, target, loaded_state):
    goal = build_target_goal(target)
    goal.name = 'level_plate_carry_goal'
    goal.position_constraints[0].constraint_region.primitives[0].dimensions = [0.001]
    constraints = Constraints()
    constraints.name = 'level_plate_carry_path'
    orientation = deepcopy(goal.orientation_constraints[0])
    orientation.absolute_x_axis_tolerance = 0.005
    orientation.absolute_y_axis_tolerance = 0.005
    orientation.absolute_z_axis_tolerance = 0.005
    constraints.orientation_constraints = [orientation]
    node.get_logger().info(
        'Planning OMPL loaded carry with orientation PATH constraints '
        '(0.005 rad per axis), attached plate geometry and a 20 s budget. '
        'A curved path is allowed. No execution.'
    )
    result = plan_to_target(
        node, goal_override=goal, start_state=loaded_state,
        path_constraints=constraints, planning_time=20.0,
    )
    return SimpleNamespace(
        start_state=deepcopy(loaded_state), solution=result.trajectory,
    )

def build_seeded_approach_goal(node, transform, *, start_state=None):
    if start_state is None:
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = (
            PlanningSceneComponents.ROBOT_STATE
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )

        state = deepcopy(_call_service(
            node,
            GetPlanningScene,
            '/get_planning_scene',
            scene_request,
        ).scene.robot_state)
    else:
        state = deepcopy(start_state)

    if state.attached_collision_objects:
        raise RuntimeError('Seeded approach requires an unloaded robot.')

    joints = state.joint_state
    if len(joints.name) != len(joints.position) or len(set(joints.name)) != len(joints.name):
        raise RuntimeError('Current joint state is incomplete or contains duplicates.')
    positions = dict(zip(joints.name, joints.position))
    # These values are NEVER sent to a controller or used directly as the goal.
    # They seed collision-checked IK for the target freshly read from TF.
    seed = {
        'joint1': -0.9975574884,
        'joint2': 1.4644051120,
        'joint3': 3.0739086998,
        'joint4': 0.0000047471,
        'joint5': 1.6094962670,
        'joint6': -0.9975315561,
    }
    if not set(seed).issubset(positions):
        raise RuntimeError('Current joint state is missing arm joints.')
    for name in ('finger_joint1', 'finger_joint2'):
        value = positions.get(name)
        if value is None or not math.isfinite(value) or value < 0.0084:
            raise RuntimeError(f'Seeded approach requires an open gripper: {name}')
    positions.update(seed)
    joints.name = list(positions)
    joints.position = list(positions.values())
    joints.velocity = []
    joints.effort = []
    joints.header.stamp.sec = 0
    joints.header.stamp.nanosec = 0
    state.is_diff = False

    request = GetPositionIK.Request()
    ik = request.ik_request
    ik.group_name = 'lite6'
    ik.robot_state = state
    ik.avoid_collisions = True
    ik.ik_link_name = 'link_eef'
    ik.pose_stamped.header = deepcopy(transform.header)
    ik.pose_stamped.pose = build_target_pose(transform)
    ik.timeout.sec = 3
    node.get_logger().info(
        'Seeded approach: solving the current TF pose with collision-checked IK. '
        'The historical pickup state is only a numerical seed. No execution.'
    )
    response = _call_service(node, GetPositionIK, '/compute_ik', request)
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        raise RuntimeError(f'Seeded approach IK failed: {response.error_code.val}')

    result = response.solution.joint_state
    if len(result.name) != len(result.position) or len(set(result.name)) != len(result.name):
        raise RuntimeError('Seeded approach IK returned an incomplete joint state.')
    solution = dict(zip(result.name, result.position))
    goal = Constraints()
    goal.name = 'seeded_approach_ik_goal'
    for name in seed:
        value = solution.get(name)
        if value is None or not math.isfinite(value):
            raise RuntimeError(f'Seeded approach IK returned an invalid joint: {name}')
        joint = JointConstraint()
        joint.joint_name = name
        joint.position = value
        joint.tolerance_above = 0.00001
        joint.tolerance_below = 0.00001
        joint.weight = 1.0
        goal.joint_constraints.append(joint)
    node.get_logger().info(
        'Approach IK candidate [rad]: '
        + ', '.join(f'{name}={solution[name]:.6f}' for name in seed)
    )
    return goal

def plan_approach(node, transform, *, strategy, start_state=None):
    if (
        start_state is not None
        and start_state.attached_collision_objects
    ):
        raise RuntimeError(
            'Approach requires an unloaded robot. '
            'Attached objects are still present.'
        )

    if strategy == 'seeded_ik':
        goal = build_seeded_approach_goal(
            node,
            transform,
            start_state=start_state,
        )

        result = plan_to_target(
            node,
            goal_override=goal,
            start_state=start_state,
        )

    elif strategy == 'pose':
        result = plan_to_target(
            node,
            transform,
            start_state=start_state,
        )

    else:
        raise ValueError(
            'precheck_approach_strategy must be pose or seeded_ik.'
        )

    if result.trajectory_start.attached_collision_objects:
        raise RuntimeError(
            'Approach requires an unloaded robot. '
            'Attached objects are still present.'
        )

    return result

def read_named_target(node, frame_name):
    original = node.get_parameter('target_frame')

    try:
        result = node.set_parameters([
            Parameter('target_frame', value=frame_name)
        ])[0]

        if not result.successful:
            raise RuntimeError(
                f'Cannot select target "{frame_name}": {result.reason}'
            )

        transform = node.read_target()

        if transform.child_frame_id != frame_name:
            raise RuntimeError(
                f'TargetReader returned "{transform.child_frame_id}" '
                f'instead of "{frame_name}".'
            )

        return transform

    finally:
        restored = node.set_parameters([original])[0]
        if not restored.successful:
            raise RuntimeError('Could not restore target_frame.')


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = TargetReader()
        node.declare_parameter('execute', False)
        node.declare_parameter('motion_goal', 'target')
        node.declare_parameter('gripper_empty_confirmed', False)
        node.declare_parameter('cartesian_phase', 'pickup')
        node.declare_parameter('pickup_frame', 'test_tool_goal')
        node.declare_parameter('lift_frame', 'test_lift_tool_goal')
        node.declare_parameter('precheck_approach_strategy', 'pose')
        node.declare_parameter('carry_frame', '')
        node.declare_parameter('carry_planner', 'cartesian')
        node.declare_parameter('carry_via_offset_deg', 15.0)
        node.declare_parameter('carry_shoulder_bend_rad', 0.0)
        node.declare_parameter('gripper_backend', 'fake')

        motion_goal = node.get_parameter('motion_goal').value
        
        if motion_goal not in (
            'target', 'home', 'diagnostic', 'cartesian', 'precheck', 'pickup_fake', 'real_check', 'approach_preview'
        ):
            raise ValueError(
                'motion_goal must be target, home, diagnostic, cartesian, precheck, pickup_fake, real_check or approach_preview.'
            )

        if (
            motion_goal in ('diagnostic', 'precheck', 'approach_preview')
            and node.get_parameter('execute').value
        ):
            raise ValueError(
                f'{motion_goal} is preview-only. Set execute:=false.'
            )

        if (
            motion_goal == 'home'
            and not node.get_parameter('gripper_empty_confirmed').value
        ):
            raise ValueError(
                'Home requires an empty gripper. '
                'Set gripper_empty_confirmed:=true after checking.'
            )
        
        if motion_goal == 'real_check':
            if node.get_parameter('execute').value:
                raise ValueError(
                    'real_check is read-only. Set execute:=false.'
                )

            from big_chemistry_motion.real_pickup import (
                require_real_environment,
            )

            require_real_environment(node)

            node.get_logger().info(
                'REAL CHECK PASSED: hardware and trajectory controller '
                'checks passed. No motion requested.'
            )
            return
        
            
        planned_world_status = check_planning_ready(node)

        if motion_goal == 'diagnostic':
            result = plan_to_target(node, diagnostic=True)

        elif motion_goal == 'home':
            result = plan_to_target(node)
            
        elif motion_goal == 'approach_preview':
            backend = node.get_parameter('gripper_backend').value

            if backend == 'fake':
                from big_chemistry_motion.fake_pickup import (
                    require_fake_environment,
                )

                limits = require_fake_environment(node)
                require_gripper_open(node)

            elif backend == 'real':
                from big_chemistry_motion.real_pickup import (
                    confirm_real_gripper_open,
                )
                from big_chemistry_motion.gripper_model import (
                    with_lite6_gripper_model_position,
                )

                limits = confirm_real_gripper_open(node)

            else:
                raise ValueError('gripper_backend must be real or fake.')

            target = node.read_target()

            scene_request = GetPlanningScene.Request()
            scene_request.components.components = (
                PlanningSceneComponents.ROBOT_STATE
                | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            )

            approach_start = deepcopy(_call_service(
                node,
                GetPlanningScene,
                '/get_planning_scene',
                scene_request,
            ).scene.robot_state)

            approach_start.is_diff = False
            
            if backend == 'real':
                approach_start = with_lite6_gripper_model_position(
                    approach_start,
                    0.0089,
                )

                node.get_logger().info(
                    'Real approach preview uses an assumed fully open '
                    'gripper model: 0.0089 m per finger. '
                    'Physical opening was confirmed by the operator; '
                    'these positions are not measured feedback.'
                )

            result = plan_approach(
                node,
                target,
                strategy=node.get_parameter(
                    'precheck_approach_strategy'
                ).value,
                start_state=approach_start,
            )

            validate_unloaded_trajectory(
                node,
                result,
                limits,
                label='APPROACH',
            )

            log_trajectory_metrics(node, result)
            preview_publisher = show_trajectory(node, result)
            check_arm_preview_start(node, result)

            if backend == 'fake':
                check_execution_start(node, result)

            node.get_logger().info(
                'APPROACH PREVIEW COMPLETE. No execution. '
                'Press Ctrl+C to close.'
            )
            rclpy.spin(node)
            return
            
        elif motion_goal in ('precheck', 'pickup_fake'):
            if motion_goal == 'pickup_fake':
                from big_chemistry_motion.fake_pickup import require_fake_environment, scene_snapshot
                require_fake_environment(node)
                original_fake_scene = scene_snapshot(node)
                if not node.get_parameter('carry_frame').value:
                    raise ValueError('pickup_fake requires carry_frame.')
            require_gripper_open(node)

            approach_target = node.read_target()
            pickup_target = read_named_target(
                node,
                node.get_parameter('pickup_frame').value,
            )

            strategy = node.get_parameter(
                'precheck_approach_strategy'
            ).value

            approach_result = plan_approach(
                node,
                approach_target,
                strategy=strategy,
            )

            approach_end = state_at_trajectory_point(
                approach_result.trajectory_start,
                approach_result.trajectory,
            )

            if motion_goal == 'pickup_fake':
                from big_chemistry_motion.fake_pickup import validate_unloaded_segment
                validate_unloaded_segment(node, approach_result, label='APPROACH')

            node.get_logger().info(
                'Checking descent from the planned approach endpoint.'
            )

            descent_response = plan_cartesian_to_target(
                node,
                pickup_target,
                start_state=approach_end,
            )

            if not diagnose_cartesian_collisions(node, descent_response):
                raise RuntimeError(
                    'Approach candidate rejected: descent collision validation failed.'
                )

            descent_result = SimpleNamespace(
                trajectory=descent_response.solution,
                trajectory_start=state_at_trajectory_point(
                    approach_end, descent_response.solution, point_index=0,
                ),
            )
            if motion_goal == 'pickup_fake':
                validate_unloaded_segment(node, descent_result, label='DESCENT')
                
            lift_target = read_named_target(
                node,
                node.get_parameter('lift_frame').value,
            )
            carry_frame = node.get_parameter('carry_frame').value
            carry_target = read_named_target(node, carry_frame) if carry_frame else None
            carry_planner = node.get_parameter('carry_planner').value
            if carry_planner not in ('cartesian', 'ompl', 'joint_probe', 'joint_via'):
                raise ValueError('carry_planner must be cartesian, ompl, joint_probe or joint_via.')

            pickup_end = state_at_trajectory_point(
                approach_end,
                descent_response.solution,
            )

            loaded_pickup_state = build_hypothetical_grasp(
                node,
                pickup_end,
            )

            node.get_logger().info(
                'Checking loaded vertical lift from the hypothetical pickup state.'
            )

            with temporary_plate_world_removal(node, loaded_pickup_state):
                check_loaded_pickup_state(node, loaded_pickup_state)

                lift_response = plan_cartesian_to_target(
                    node,
                    lift_target,
                    start_state=loaded_pickup_state,
                )

                validate_loaded_trajectory(
                    node, lift_response, loaded_pickup_state,
                    label='LOADED LIFT', allow_initial_support=True,
                )

                if carry_target is not None:
                    lifted_state = state_at_trajectory_point(
                        loaded_pickup_state, lift_response.solution,
                    )
                    node.get_logger().info(
                        'Checking loaded carry from the planned lift endpoint.'
                    )
                    if carry_planner == 'ompl':
                        carry_response = plan_level_carry(node, carry_target, lifted_state)
                    elif carry_planner == 'joint_probe':
                        carry_response = plan_joint_carry_probe(node, carry_target, lifted_state)
                    elif carry_planner == 'joint_via':
                        carry_response = plan_joint_carry_probe(
                            node, carry_target, lifted_state,
                            via_offset_deg=node.get_parameter('carry_via_offset_deg').value,
                            shoulder_bend_rad=node.get_parameter('carry_shoulder_bend_rad').value,
                        )
                    else:
                        carry_response = plan_cartesian_to_target(
                            node, carry_target, start_state=lifted_state,
                        )
                    validate_loaded_trajectory(
                        node, carry_response, lifted_state, label='LOADED CARRY',
                    )

            node.get_logger().info(
                'PRECHECK PASSED: approach, descent and loaded lift'
                + (' and carry' if carry_target is not None else '')
                + ' are plannable. '
                + ('Four phase previews follow.' if motion_goal == 'pickup_fake'
                   else 'RViz will show the approach trajectory only.')
            )

            result = approach_result

            if motion_goal == 'pickup_fake':
                from big_chemistry_motion.fake_pickup import review_and_run_fake_pickup
                segments = [
                    ('APPROACH', approach_result),
                    ('DESCENT', descent_result),
                    ('LOADED LIFT', SimpleNamespace(
                        trajectory=lift_response.solution,
                        trajectory_start=deepcopy(loaded_pickup_state),
                    )),
                    ('LOADED CARRY', SimpleNamespace(
                        trajectory=carry_response.solution,
                        trajectory_start=deepcopy(lifted_state),
                    )),
                ]
                review_and_run_fake_pickup(
                    node, segments, planned_world_status, original_fake_scene,
                )
                return

        elif motion_goal == 'cartesian':
            target = node.read_target()
    
            phase = node.get_parameter('cartesian_phase').value

            if phase not in ('pickup', 'lift'):
                raise ValueError('cartesian_phase must be pickup or lift.')

            if phase == 'pickup':
                require_gripper_open(node)

            
            response = plan_cartesian_to_target(node, target)

            if not diagnose_cartesian_collisions(node, response):
                raise RuntimeError(
                    'Cartesian trajectory failed collision validation. '
                    'Execution blocked.'
                )
            
            start_state = deepcopy(response.start_state)
            joint_state = start_state.joint_state
            positions = dict(zip(joint_state.name, joint_state.position))

            trajectory = response.solution.joint_trajectory
            positions.update(zip(
                trajectory.joint_names,
                trajectory.points[0].positions,
            ))

            joint_state.name = list(positions.keys())
            joint_state.position = list(positions.values())
            joint_state.velocity = []
            joint_state.effort = []

            result = SimpleNamespace(
                trajectory=response.solution,
                trajectory_start=start_state,
            )

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
