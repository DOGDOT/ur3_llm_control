import os
import re
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('ur3_llm_control')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    ur_moveit_share = get_package_share_directory('ur_moveit_config')

    world_path = os.path.join(pkg_share, 'worlds', 'ur3_cubes_camera.world')
    xacro_file = os.path.join(pkg_share, 'urdf', 'ur3_with_gripper.urdf.xacro')

    # 1. Khởi chạy Gazebo Classic
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_path, 'verbose': 'true', 'gui': 'true'}.items()
    )

    # 2. Xử lý xacro và loại bỏ comment gây lỗi CLI parser
    doc = xacro.process_file(xacro_file)
    raw_xml = doc.toxml()
    cleaned_xml = re.sub(r'<!--.*?-->', '', raw_xml, flags=re.DOTALL)

    temp_urdf_path = '/tmp/generated_ur3_with_gripper.urdf'
    with open(temp_urdf_path, 'w') as f:
        f.write(cleaned_xml)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': cleaned_xml,
            'use_sim_time': True
        }]
    )

    # 3. Spawn UR3 mang Gripper vào Gazebo
    spawn_ur3 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_ur3_robot',
        output='screen',
        arguments=[
            '-file', temp_urdf_path,
            '-entity', 'ur3_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ]
    )

    # 4. Spawners cho Controllers
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    load_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen'
    )

    delay_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_ur3,
            on_exit=[load_joint_state_broadcaster, load_arm_controller, load_gripper_controller],
        )
    )

    # 5. Khởi chạy MoveIt 2 và RViz2 tích hợp sẵn (chờ sau khi nạp controllers 4 giây)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_moveit_share, 'launch', 'ur_moveit.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3',
            'launch_rviz': 'true',
            'use_sim_time': 'true'
        }.items()
    )

    delayed_moveit = TimerAction(
        period=5.0,
        actions=[moveit_launch]
    )

    # 6. Khởi chạy Camera Perception Node (tự động nhận diện block và zone)
    camera_perception = Node(
        package='ur3_llm_control',
        executable='camera_perception',
        name='camera_perception_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    delayed_camera = TimerAction(
        period=4.0,
        actions=[camera_perception]
    )

    # Cấu hình plugin path cho Gazebo LinkAttacher
    attacher_share = get_package_share_directory('ros2_linkattacher')
    attacher_lib_path = os.path.abspath(os.path.join(attacher_share, '..', '..', 'lib'))
    current_plugin_path = os.environ.get('GAZEBO_PLUGIN_PATH', '')
    new_plugin_path = f"{current_plugin_path}:{attacher_lib_path}" if current_plugin_path else attacher_lib_path
    set_gazebo_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value=new_plugin_path
    )

    return LaunchDescription([
        set_gazebo_plugin_path,
        gazebo,
        robot_state_publisher,
        spawn_ur3,
        delay_controllers,
        delayed_moveit,
        delayed_camera
    ])