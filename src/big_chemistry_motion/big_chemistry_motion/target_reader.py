import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class TargetReader(Node):
    def __init__(self):
        super().__init__('target_reader')

        self.declare_parameter('reference_frame', 'link_base')
        self.declare_parameter('target_frame', 'test_pickup')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def read_target(self):
        reference_frame = self.get_parameter('reference_frame').value
        target_frame = self.get_parameter('target_frame').value

        for frame in (reference_frame, target_frame):
            if not isinstance(frame, str) or not frame.strip():
                raise ValueError('Frame names must be non-empty strings.')

        self.get_logger().info(
            f'Waiting for target "{target_frame}" '
            f'relative to "{reference_frame}"...'
        )

        deadline = time.monotonic() + 10.0
        last_error = 'No transform received.'

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                transform = self.tf_buffer.lookup_transform(
                    reference_frame,
                    target_frame,
                    Time(),
                )
            except TransformException as error:
                last_error = str(error)
                continue

            position = transform.transform.translation
            rotation = transform.transform.rotation

            self.get_logger().info(
                f'Position [m]: '
                f'x={position.x:.3f}, '
                f'y={position.y:.3f}, '
                f'z={position.z:.3f}'
            )
            self.get_logger().info(
                f'Orientation [xyzw]: '
                f'[{rotation.x:.3f}, {rotation.y:.3f}, '
                f'{rotation.z:.3f}, {rotation.w:.3f}]'
            )

            return transform

        raise RuntimeError(
            f'Could not resolve "{target_frame}" '
            f'relative to "{reference_frame}": {last_error}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = TargetReader()
        node.read_target()
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