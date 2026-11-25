from copy import deepcopy
import math


def with_lite6_gripper_model_position(reference_state, position):
    """Copy a robot state and set assumed gripper model positions."""
    if not math.isfinite(position) or not 0.0 <= position <= 0.0089:
        raise ValueError(
            'Gripper model position must be within 0–0.0089 m.'
        )

    state = deepcopy(reference_state)
    joints = state.joint_state

    if len(joints.name) != len(joints.position):
        raise RuntimeError('Reference joint names and positions do not match.')

    if len(set(joints.name)) != len(joints.name):
        raise RuntimeError('Reference state contains duplicate joint names.')

    positions = dict(zip(joints.name, joints.position))
    positions['finger_joint1'] = position
    positions['finger_joint2'] = position

    joints.name = list(positions)
    joints.position = list(positions.values())
    joints.velocity = []
    joints.effort = []
    joints.header.stamp.sec = 0
    joints.header.stamp.nanosec = 0

    return state