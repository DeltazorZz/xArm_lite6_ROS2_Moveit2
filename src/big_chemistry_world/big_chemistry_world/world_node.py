"""World configuration and simulated Position Check.

Single-threaded executor only. Status is advisory: external planners/executors
must respect it; this node does not interlock plan_to_target or robot motion.
Tag poses are assumed to be preprocessed in world_frame. Capture timestamp,
stability, displacement limits, and multi-device calibration are future work.
"""
from pathlib import Path
import math
import json
import time
from rclpy.clock import Clock, ClockType
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from big_chemistry_world.scene_objects import create_collision_object
from moveit_msgs.msg import PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from copy import deepcopy
import tf2_geometry_msgs
from tf2_ros import Buffer, TransformException
from std_srvs.srv import Trigger

class WorldNode(Node):
    def __init__(self):
        super().__init__('big_chemistry_world')

        default_config = (Path(get_package_share_directory('big_chemistry_world')) / 'config' / 'lab_world.yaml')
        self.declare_parameter('config_path', str(default_config))

        self.tf_broadcaster = StaticTransformBroadcaster(self)

        self.declare_parameter('collision_reference_frame', 'world')
        self.config_tf_buffer = Buffer()

        self.scene_ready = False
        self.config_transforms = []
        self.pending_scene_update = None

        self.scene_update_client = self.create_client(
            ApplyPlanningScene,
            '/apply_planning_scene',
        )

        self.latest_tag_pose = None

        self.tag_pose_subscription = self.create_subscription(
            PoseStamped,
            '/test_tag/pose',
            self.on_tag_pose,
            10,
        )

        self.position_check_requested = False
        self.position_check_id = 0
        self.position_check_state = 'IDLE'
        self.position_check_detail = 'No position check has been started.'
        self.position_check_deadline = None
        self.declare_parameter('measurement_timeout_sec', 10.0)
        self.declare_parameter('scene_update_timeout_sec', 10.0)
        self.measurement_timeout_sec = float(self.get_parameter('measurement_timeout_sec').value)
        self.scene_update_timeout_sec = float(self.get_parameter('scene_update_timeout_sec').value)
        for value in (self.measurement_timeout_sec, self.scene_update_timeout_sec):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Position check timeouts must be finite and positive.')
        self.position_check_status_service = self.create_service(
            Trigger,
            '/big_chemistry_world/position_check_status',
            self.on_position_check_status,
        )
        # Timeout handling also runs when ROS simulation time is paused.
        self.timeout_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.position_check_timer = self.create_timer(
            0.1, self.check_position_check_timeout, clock=self.timeout_clock,
        )

        self.position_check_service = self.create_service(
            Trigger,
            '/big_chemistry_world/position_check',
            self.on_position_check,
        )

    def _set_check_state(self, state, detail):
        self.position_check_state = state
        self.position_check_detail = detail
        self.get_logger().info(f'Position check #{self.position_check_id}: {state}: {detail}')

    def on_position_check(self, request, response):
        self.check_position_check_timeout()
        if not self.scene_ready:
            response.success = False
            response.message = 'Scene is not confirmed. Restart MoveIt and world_node.'
            return response
        if self.position_check_state in ('WAITING_FOR_MEASUREMENT', 'APPLYING'):
            response.success = False
            response.message = 'A position check is already pending.'
            return response

        self.position_check_id += 1
        self.position_check_requested = True
        self.position_check_deadline = time.monotonic() + self.measurement_timeout_sec
        self._set_check_state(
            'WAITING_FOR_MEASUREMENT',
            'Waiting for a valid message received after this request.',
        )
        response.success = True
        response.message = (
            f'Position check #{self.position_check_id} started, not completed. '
            'Query /big_chemistry_world/position_check_status for the result.'
        )
        return response

    def on_position_check_status(self, request, response):
        self.check_position_check_timeout()
        # success means the status query succeeded, NOT that the check succeeded.
        response.success = True
        response.message = json.dumps({
            'check_id': self.position_check_id,
            'state': self.position_check_state,
            'detail': self.position_check_detail,
            'scene_confirmed': self.scene_ready,
            'planning_ready': (
                self.scene_ready
                and self.position_check_state not in (
                    'WAITING_FOR_MEASUREMENT', 'APPLYING'
                )
            ),
        })
        return response

    def _fail_position_check(self, detail, scene_unconfirmed=False):
        self.position_check_requested = False
        self.position_check_deadline = None
        if scene_unconfirmed:
            self.scene_ready = False
        self._set_check_state('FAILED', detail)

    def check_position_check_timeout(self):
        deadline = self.position_check_deadline
        if deadline is None or time.monotonic() < deadline:
            return
        if self.position_check_state == 'WAITING_FOR_MEASUREMENT':
            self._fail_position_check(
                'Measurement timeout. Previous scene retained; a new check may be started.'
            )
        elif self.position_check_state == 'APPLYING':
            pending = self.pending_scene_update
            # Invalidate ownership BEFORE cancelling: late callbacks must do nothing.
            self.pending_scene_update = None
            self._fail_position_check(
                'MoveIt response timeout. Scene state is unconfirmed. '
                'Do not plan; restart MoveIt and world_node.',
                scene_unconfirmed=True,
            )
            if pending is not None:
                # Local cancellation cannot undo an already applied server request.
                pending[0].cancel()

    def on_tag_pose(self, message):
        self.check_position_check_timeout()
        if not hasattr(self, 'config'):
            return
        expected_frame = self.config['world_frame']
        if message.header.frame_id != expected_frame:
            self.get_logger().warning(
                f'Tag pose rejected: expected "{expected_frame}", '
                f'got "{message.header.frame_id}".'
            )
            return
        p, q = message.pose.position, message.pose.orientation
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            self.get_logger().warning('Tag pose rejected: non-finite values.')
            return
        norm = math.hypot(q.x, q.y, q.z, q.w)
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.01:
            self.get_logger().warning('Tag pose rejected: quaternion must have unit length.')
            return
        accepted = deepcopy(message)
        rotation = accepted.pose.orientation
        rotation.x /= norm
        rotation.y /= norm
        rotation.z /= norm
        rotation.w /= norm
        self.latest_tag_pose = accepted
        # This simulation gate checks receipt order, not camera capture timestamps.
        if not self.position_check_requested:
            return
        self.position_check_requested = False
        self.request_tag_update(accepted)

    def request_tag_update(self, pose):
        if self.position_check_state != 'WAITING_FOR_MEASUREMENT':
            return
        self.check_position_check_timeout()
        if self.position_check_state != 'WAITING_FOR_MEASUREMENT':
            return
        if not self.scene_ready:
            self._fail_position_check('Scene is not ready.')
            return
        if not self.scene_update_client.service_is_ready():
            self._fail_position_check(
                'MoveIt service unavailable. Re-establish the scene before planning.',
                scene_unconfirmed=True,
            )
            return

        try:
            candidate_transforms = deepcopy(self.config_transforms)
            tags = [t for t in candidate_transforms if t.child_frame_id == 'test_tag']
            if len(tags) != 1:
                raise ValueError('Expected exactly one configured test_tag frame.')
            tag = tags[0]
            if tag.header.frame_id != pose.header.frame_id:
                raise ValueError('Tag measurement frame differs from configured parent.')
            tag.header.stamp = self.get_clock().now().to_msg()
            tag.transform.translation.x = pose.pose.position.x
            tag.transform.translation.y = pose.pose.position.y
            tag.transform.translation.z = pose.pose.position.z
            tag.transform.rotation = deepcopy(pose.pose.orientation)
            candidate_buffer = Buffer()
            for transform in candidate_transforms:
                candidate_buffer.set_transform_static(transform, 'world_candidate')
            objects = self.prepare_scene_objects(candidate_buffer)
            request = ApplyPlanningScene.Request()
            request.scene.is_diff = True
            request.scene.robot_state.is_diff = True
            request.scene.world.collision_objects = objects
        except Exception as error:
            self._fail_position_check(f'Could not prepare scene update: {error}')
            return

        self.position_check_deadline = time.monotonic() + self.scene_update_timeout_sec
        self._set_check_state('APPLYING', 'Waiting for MoveIt acknowledgement; do not plan yet.')
        try:
            future = self.scene_update_client.call_async(request)
        except Exception as error:
            self._fail_position_check(
                f'Could not send update: {error}. Scene unconfirmed; restart MoveIt and world_node.',
                scene_unconfirmed=True,
            )
            return
        self.pending_scene_update = (
            future, candidate_transforms, candidate_buffer, self.position_check_id
        )
        future.add_done_callback(self.finish_tag_update)

    def finish_tag_update(self, future):
        self.check_position_check_timeout()
        pending = self.pending_scene_update
        if pending is None or pending[0] is not future:
            return
        _, transforms, buffer, check_id = pending
        if check_id != self.position_check_id or self.position_check_state != 'APPLYING':
            return
        try:
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError('MoveIt did not confirm the scene update.')
            # Calibration snapshots: publish only after scene acknowledgement.
            self.tf_broadcaster.sendTransform(transforms)
            self.config_tf_buffer = buffer
            self.config_transforms = transforms
        except Exception as error:
            self._fail_position_check(
                f'Update failed: {error}. Scene unconfirmed; restart MoveIt and world_node.',
                scene_unconfirmed=True,
            )
        else:
            self.position_check_deadline = None
            self._set_check_state(
                'SUCCEEDED',
                'Tag update applied: collision objects and target frames updated.',
            )
        finally:
            self.pending_scene_update = None

    def load_config(self):
        config_path = Path(
            self.get_parameter('config_path').value).expanduser()

        with config_path.open(encoding='utf-8') as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Configuration must be YAML mapping.")

        if type(config.get('schema_version')) is not int:
            raise ValueError(f"schema_version must be an integer.")

        if config['schema_version'] != 1:
            raise ValueError("Only schema_version 1 is supported.")

        world_frame = config.get('world_frame')
        if not isinstance(world_frame, str) or not world_frame:
            raise ValueError('world_frame must be a non_empty string.')

        for field in ('frames', 'collision_objects'):
            if not isinstance(config.get(field), list):
                raise ValueError(f'{field} must be a list.')

        self.config = config

        self.get_logger().info(f'Configuration loaded: {config_path}')
        self.get_logger().info(f'World frame: {world_frame}')
        self.get_logger().info(f"Configured frames: {len(config['frames'])}")
        self.get_logger().info(
            f"Configured collision objects: "
            f"{len(config['collision_objects'])}"
        )

    def publish_static_frames(self):
        transforms = []
        children = set()

        for frame in self.config['frames']:
            if not isinstance(frame, dict):
                raise ValueError('Each frame must be a YAML mapping.')

            parent = frame.get('parent')
            child = frame.get('child')

            for name in (parent, child):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError('Frame names must be non-empty strings.')

            if parent == child:
                raise ValueError(f'Frame cannot be its own parent: {child}')

            if child in children:
                raise ValueError(f'Duplicate child frame: {child}')

            children.add(child)

            for field in ('translation', 'rotation_rpy'):
                values = frame.get(field)

                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError(f'{child}: {field} needs three numbers.')

                if not all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in values
                ):
                    raise ValueError(
                        f'{child}: {field} must contain finite numbers.'
                    )

            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent
            transform.child_frame_id = child

            x, y, z = frame['translation']
            transform.transform.translation.x = float(x)
            transform.transform.translation.y = float(y)
            transform.transform.translation.z = float(z)

            roll, pitch, yaw = frame['rotation_rpy']
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

            rotation = transform.transform.rotation
            rotation.x = sr * cp * cy - cr * sp * sy
            rotation.y = cr * sp * cy + sr * cp * sy
            rotation.z = cr * cp * sy - sr * sp * cy
            rotation.w = cr * cp * cy + sr * sp * sy

            transforms.append(transform)

        self.config_transforms = deepcopy(transforms)

        for transform in transforms:
            self.config_tf_buffer.set_transform_static(
                transform,
                'big_chemistry_world_config',
            )

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

        for transform in transforms:
            self.get_logger().info(
                f'Static TF published: '
                f'{transform.header.frame_id} -> {transform.child_frame_id}'
            )

    def apply_collision_objects(self):
        if not self.collision_objects:
            self.get_logger().info('No collision objects to apply.')
            return

        client = self.create_client(
            ApplyPlanningScene,
            '/apply_planning_scene',
        )

        try:
            self.get_logger().info(
                'Waiting for MoveIt planning scene service...'
            )

            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    'MoveIt service unavailable. Start the MoveIt launch first.'
                )

            scene = PlanningScene()
            scene.is_diff = True
            scene.robot_state.is_diff = True
            scene.world.collision_objects = self.prepare_scene_objects()

            request = ApplyPlanningScene.Request()
            request.scene = scene

            future = client.call_async(request)

            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=10.0,
            )

            if not future.done():
                future.cancel()
                raise RuntimeError(
                    'Timed out waiting for MoveIt. Scene state is unconfirmed.'
                )

            response = future.result()

            if response is None or not response.success:
                raise RuntimeError(
                    'MoveIt could not apply the collision objects.'
                )

            self.get_logger().info(
                f'MoveIt accepted {len(self.collision_objects)} '
                f'collision object(s).'
            )
        finally:
            self.destroy_client(client)

    def prepare_scene_objects(self, transform_buffer=None):
        if transform_buffer is None:
            transform_buffer = self.config_tf_buffer
        target_frame = self.get_parameter(
            'collision_reference_frame'
        ).value

        if not isinstance(target_frame, str) or not target_frame.strip():
            raise ValueError(
                'collision_reference_frame must be a non-empty string.'
            )

        objects = []

        for original in self.collision_objects:
            converted = deepcopy(original)
            converted_poses = []

            for pose in original.primitive_poses:
                stamped_pose = PoseStamped()
                stamped_pose.header.frame_id = original.header.frame_id
                stamped_pose.pose = deepcopy(pose)

                if original.header.frame_id == target_frame:
                    converted_poses.append(deepcopy(pose))
                    continue

                try:
                    transformed_pose = transform_buffer.transform(
                        stamped_pose,
                        target_frame,
                    )
                except TransformException as error:
                    raise RuntimeError(
                        f'Cannot transform object "{original.id}" '
                        f'from "{original.header.frame_id}" '
                        f'to "{target_frame}" using configured frames: {error}'
                    ) from error

                converted_poses.append(transformed_pose.pose)

            converted.header.frame_id = target_frame
            converted.primitive_poses = converted_poses
            objects.append(converted)

            self.get_logger().info(
                f'Object "{original.id}" prepared in frame "{target_frame}".'
            )

        return objects

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = WorldNode()

        try:
            node.load_config()
            node.collision_objects = [
                create_collision_object(config)
                for config in node.config['collision_objects']
            ]

            object_ids = [obj.id for obj in node.collision_objects]
            if len(object_ids) != len(set(object_ids)):
                raise ValueError('Collision object IDs must be unique.')

            node.publish_static_frames()

            node.get_logger().info(
                f'Prepared collision objects: {len(node.collision_objects)}'
            )

            node.apply_collision_objects()

            node.scene_ready = True

        except (OSError, ValueError, yaml.YAMLError, RuntimeError) as error:
            node.get_logger().error(
                f'Could not initialize world: {error}'
            )
            raise SystemExit(1) from error

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
