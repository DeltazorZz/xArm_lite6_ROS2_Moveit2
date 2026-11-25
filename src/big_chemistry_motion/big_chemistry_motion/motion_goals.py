import math
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive

HOME_JOINT_DEGREES = {
    'joint1': 0.0,
    'joint2': 9.9,
    'joint3': 31.8,
    'joint4': 0.0,
    'joint5': 21.9,
    'joint6': 0.0,
}


def build_home_goal():
    goal = Constraints()
    goal.name = 'home'

    for joint_name, angle_degrees in HOME_JOINT_DEGREES.items():
        joint = JointConstraint()
        joint.joint_name = joint_name
        joint.position = math.radians(angle_degrees)
        joint.tolerance_above = 0.005
        joint.tolerance_below = 0.005
        joint.weight = 1.0

        goal.joint_constraints.append(joint)

    return goal

def build_target_goal(transform):
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
    orientation.absolute_x_axis_tolerance = 0.005
    orientation.absolute_y_axis_tolerance = 0.005
    orientation.absolute_z_axis_tolerance = 0.005
    orientation.weight = 1.0

    goal = Constraints()
    goal.name = 'device_relative_target'
    goal.position_constraints = [position]
    goal.orientation_constraints = [orientation]
    
    return goal

def build_diagnostic_joint_goal():
    # Az előző, ütközésellenőrzött IK-lekérdezés eredménye.
    # Csak az akkori test_tool_goal vizsgálatához.
    joint_positions = {
        'joint1': -0.9975575080382395,
        'joint2': 1.1376874331830829,
        'joint3': 2.479148684399614,
        'joint4': 0.000005177378247140018,
        'joint5': 1.3414539053600996,
        'joint6': -0.9975336626521848,
    }

    goal = Constraints()
    goal.name = 'diagnostic_joint_goal'

    for joint_name, position in joint_positions.items():
        joint = JointConstraint()
        joint.joint_name = joint_name
        joint.position = position
        joint.tolerance_above = 0.001
        joint.tolerance_below = 0.001
        joint.weight = 1.0
        goal.joint_constraints.append(joint)

    return goal


def build_target_pose(transform):
    pose = Pose()

    pose.position.x = transform.transform.translation.x
    pose.position.y = transform.transform.translation.y
    pose.position.z = transform.transform.translation.z

    pose.orientation.x = transform.transform.rotation.x
    pose.orientation.y = transform.transform.rotation.y
    pose.orientation.z = transform.transform.rotation.z
    pose.orientation.w = transform.transform.rotation.w

    return pose