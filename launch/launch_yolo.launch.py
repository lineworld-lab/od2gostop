from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description = 'Use simulation (Gazebo) clock if true'),
        Node(package='yolo_ros', executable='2_polo_pt.py', output='screen'),
    ])