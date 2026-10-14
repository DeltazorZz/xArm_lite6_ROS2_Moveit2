from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = (
        Path(get_package_share_directory('big_chemistry_world'))
        / 'config'
        / 'lab_world.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_path',
            default_value=str(default_config),
            description='Path to the laboratory world configuration.',
        ),
        Node(
            package='big_chemistry_world',
            executable='world_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
            }],
        ),
    ])