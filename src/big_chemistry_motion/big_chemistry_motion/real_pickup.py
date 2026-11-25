from big_chemistry_motion.hardware_environment import (
    read_arm_joint_limits,
    require_hardware_environment,
    require_spline_interpolation,
)
from big_chemistry_motion.motion_checks import (
    check_controller_ready,
    check_execution_interfaces,
)
from big_chemistry_motion.gripper_control import (
    check_real_gripper_services,
)


def require_real_environment(node):
    if node.get_parameter('gripper_backend').value != 'real':
        raise RuntimeError('Real pickup requires gripper_backend:=real.')

    require_hardware_environment(node, expected_backend='real')
    limits = read_arm_joint_limits(node)

    check_controller_ready(node)
    require_spline_interpolation(node)
    check_execution_interfaces(node)
    check_real_gripper_services(node)

    node.get_logger().info(
        'Real environment checks passed and arm joint limits read. '
        'No motion requested.'
    )

    return limits

def confirm_real_gripper_open(node):
    """Require an operator's visual confirmation; no gripper command."""
    limits = require_real_environment(node)

    try:
        answer = input(
            'Visually check that the REAL gripper is fully open.\n'
            'Type OPEN to confirm; anything else cancels: '
        ).strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise RuntimeError(
            'Real gripper opening was not confirmed.'
        ) from error

    if answer != 'OPEN':
        raise RuntimeError(
            'Real gripper opening was not confirmed.'
        )

    node.get_logger().info(
        'Operator visually confirmed the real gripper is fully open. '
        'No measured finger position is available from this check.'
    )
    
    return limits