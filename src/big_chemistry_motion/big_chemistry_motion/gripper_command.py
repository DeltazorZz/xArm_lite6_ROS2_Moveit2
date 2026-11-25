import rclpy
from rclpy.node import Node

from big_chemistry_motion.gripper_control import (
    open_gripper,
    close_gripper,
    stop_gripper,
)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Node('gripper_command')
        node.declare_parameter('gripper_backend', '')
        node.declare_parameter('command', '')
        node.declare_parameter('fake_position', -1.0)

        backend = node.get_parameter('gripper_backend').value
        command = node.get_parameter('command').value
        position = node.get_parameter('fake_position').value

        if backend not in ('real', 'fake'):
            raise ValueError('Select gripper_backend:=real or fake.')

        if command not in ('open', 'close', 'stop'):
            raise ValueError('Select command:=open, close or stop.')

        if position != -1.0 and not (
            backend == 'fake' and command == 'close'
        ):
            raise ValueError(
                'fake_position is only valid for fake closing.'
            )

        if backend == 'fake' and command == 'close':
            if not 0.0 <= position <= 0.0089:
                raise ValueError(
                    'Fake closing requires fake_position within 0–0.0089 m.'
                )

        # Stop must not wait for interactive confirmation.
        if backend == 'real' and command != 'stop':
            answer = input(
                f'Real gripper command: {command}.\n'
                'Type EXECUTE to continue; anything else cancels: '
            ).strip()

            if answer != 'EXECUTE':
                node.get_logger().info('Cancelled. No command sent.')
                return

        if command == 'open':
            open_gripper(node)
        elif command == 'close':
            if backend == 'fake':
                close_gripper(node, fake_position=position)
            else:
                close_gripper(node)
        else:
            stop_gripper(node)

    except (ValueError, RuntimeError) as error:
        if node is not None:
            node.get_logger().error(str(error))
        raise SystemExit(1) from error
    except (KeyboardInterrupt, EOFError):
        if node is not None:
            node.get_logger().warning(
                'Interrupted. Any command already sent may still take effect; '
                'gripper state is not confirmed.'
            )
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()