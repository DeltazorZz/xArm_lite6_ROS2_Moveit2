import math

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


def read_vector(config, field, length):
    values = config.get(field)

    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f'{field} must contain {length} numbers.')

    if not all(
        type(value) in (int, float) and math.isfinite(value)
        for value in values
    ):
        raise ValueError(f'{field} must contain finite numbers.')

    return [float(value) for value in values]


def create_collision_object(config):
    if not isinstance(config, dict):
        raise ValueError('Each collision object must be a YAML mapping.')

    for field in ('id', 'frame_id'):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} must be a non-empty string.')

    if config.get('type') != 'box':
        raise ValueError('Only box collision objects are supported for now.')

    dimensions = read_vector(config, 'dimensions', 3)
    position = read_vector(config, 'position', 3)
    orientation = read_vector(config, 'orientation_xyzw', 4)

    if any(value <= 0 for value in dimensions):
        raise ValueError('Box dimensions must be positive.')

    quaternion_length = math.sqrt(sum(value ** 2 for value in orientation))
    if not math.isclose(quaternion_length, 1.0, abs_tol=1e-6):
        raise ValueError('orientation_xyzw must be a unit quaternion.')

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = dimensions

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = orientation

    collision_object = CollisionObject()
    collision_object.header.frame_id = config['frame_id']
    collision_object.id = config['id']
    collision_object.primitives = [primitive]
    collision_object.primitive_poses = [pose]
    collision_object.operation = CollisionObject.ADD

    return collision_object