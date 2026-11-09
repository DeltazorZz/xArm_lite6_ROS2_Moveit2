import math
import time

import rclpy
from moveit_msgs.msg import DisplayTrajectory
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

def log_trajectory_metrics(node, result): 
    trajectory = result.trajectory.joint_trajectory
    names = list(trajectory.joint_names)
    points = trajectory.points

    if not names or not points: 
        raise ValueError('Cannot measure an empty trajectory.')
    
    for point in points: 
        if len(point.positions) != len(names):
            raise ValueError('Trajectory joint positions are incompleate.')
        
        if not all(math.isfinite(value) for value in point.positions):
            raise ValueError('Trajectory contains non-finite joint positions.')
        
    start_state = result.trajectory_start.joint_state
    start_positions = dict(zip(start_state.name, start_state.position))
    
    logger = node.get_logger()
    logger.info('--- Trajectory metrics ---')

    missing = [name for name in names if name not in start_positions]
    if missing: 
        logger.warning(
            f'Start state is missing joints: {", ".join(missing)}'
        )
        
    for index, name in enumerate(names):
        first_deg = math.degrees(points[0].positions[index])
        
        if name in start_positions:
            start_deg = math.degrees(start_positions[name])
            logger.info(
                f'{name}: start={start_deg:.3f} deg; '
                f'first trajectory point={first_deg:.3f} deg'
            )
            
        else: 
            logger.info(
                f'{name}: start=unknown, '
                f'first trajectory point={first_deg:.3f} deg'
            )

    last_time = points[-1].time_from_start
    duration = last_time.sec + last_time.nanosec * 1e-9
    
    travel_per_joint = [0.0] * len(names)
    
    
    for previous, current in zip(points, points[1:]):
        for index in range(len(names)):
            travel_per_joint[index] += abs(
                current.positions[index] - previous.positions[index]
            )
            
    total_travel = sum(travel_per_joint)
    
    
    logger.info(f'Planned trajectory duration: {duration:.3f} s')
    logger.info(
        f'Total joint travel: {total_travel:.3f} rad '
        f'({math.degrees(total_travel):.1f} deg summed across joints)'
    )

    for name, travel in zip(names, travel_per_joint):
        logger.info(
            f'{name}: accummulated travel={math.degrees(travel):.1f} deg.'
        )
        
    times = [
        point.time_from_start.sec
        + point.time_from_start.nanosec * 1e-9
        for point in points
    ]

    intervals = [
        later - earlier
        for earlier, later in zip(times, times[1:])
    ]

    if not intervals or any(
        not math.isfinite(dt) or dt <= 0.0
        for dt in intervals
    ):
        raise ValueError('Invalid trajectory time intervals.')

    logger.info(
        f'Trajectory point spacing: '
        f'min={min(intervals):.4f} s, '
        f'max={max(intervals):.4f} s'
    )

    for index, name in enumerate(names):
        segment_peak = max(
            abs(current.positions[index] - previous.positions[index]) / dt
            for previous, current, dt in zip(
                points, points[1:], intervals
            )
        )

        if all(len(point.velocities) == len(names) for point in points):
            velocities = [point.velocities[index] for point in points]

            if not all(math.isfinite(value) for value in velocities):
                raise ValueError(f'Invalid planned velocity: {name}')

            stored_peak = max(abs(value) for value in velocities)
            stored_text = f'{math.degrees(stored_peak):.2f} deg/s'
        else:
            stored_text = 'not available'

        logger.info(
            f'{name}: stored peak velocity={stored_text}; '
            f'peak segment-average velocity='
            f'{math.degrees(segment_peak):.2f} deg/s'
        )    
        
        
    logger.info('--- End of trajectory metrics ---')
    
def show_trajectory(node, result):
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    publisher = node.create_publisher(
        DisplayTrajectory,
        '/display_planned_path',
        qos,
    )

    node.get_logger().info('Waiting for a trajectory display subscriber...')

    deadline = time.monotonic() + 10.0

    while rclpy.ok() and publisher.get_subscription_count() == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                'No trajectory display subscriber found. '
                'Check the RViz Planned Path display.'
            )

        rclpy.spin_once(node, timeout_sec=0.1)

    if not rclpy.ok():
        return

    display = DisplayTrajectory()
    display.trajectory_start = result.trajectory_start
    display.trajectory = [result.trajectory]

    publisher.publish(display)

    node.get_logger().info(
        'Trajectory preview published. No execution.'
    )

    return publisher