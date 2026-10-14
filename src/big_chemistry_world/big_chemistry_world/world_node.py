from pathlib import Path
import math
import rclpy
import yaml 
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from big_chemistry_world.scene_objects import create_collision_object
from moveit_msgs.msg import PlanningScene
from moveit_msgs.srv import ApplyPlanningScene

class WorldNode(Node):
    def __init__(self):
        super().__init__('big_chemistry_world')
        
        default_config = (Path(get_package_share_directory('big_chemistry_world')) / 'config' / 'lab_world.yaml')
        self.declare_parameter('config_path', str(default_config))
        
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
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
                raise ValueError('Each frame must be a YAML mapping. ')
            
            parent = frame.get('parent')
            child = frame.get('child')
            
            for name in (parent, child): 
                if not  isinstance(name, str) or not name.strip():
                    raise ValueError("Frame name must be non-empty string. ")
                
            if parent == child: 
                raise ValueError(f'Frame cannot be its own parent: {child}')
            
            if child in children:
                raise ValueError(f'Duplicate child frame: {child}')
        
            children.add(child)
            
            for field in ('translation', 'rotation_rpy'):
                values = frame.get(field)
    
            if not  isinstance(values, list) or len(values) != 3:
                raise ValueError(f'{child}: {field} needs three numbers. ')
            
            if not all(
                type(value) in (int, float) and math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    f'{child}: {field} must contain finite numbers. '
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
            rotation.x = sr * cp * cy * - cr* sp * sy
            rotation.y = cr * sp * cy + sr * cp * sy
            rotation.z = cr * cp * sy - sr * sp * cy
            rotation.w = cr * cp * cy + sr * sp * sy
            
            transforms.append(transform)
            
        if transform:
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
            scene.world.collision_objects = self.collision_objects

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
            
        except (OSError, ValueError, yaml.YAMLError, RuntimeError) as error: 
            node.get_logger().error(
                f'Cloud not initialize world: {error}'
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