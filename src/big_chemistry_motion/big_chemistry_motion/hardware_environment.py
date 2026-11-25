import xml.etree.ElementTree as ET
import math

def hardware_plugins_from_description(description):
    try:
        root = ET.fromstring(description)
    except ET.ParseError as error:
        raise RuntimeError('Invalid robot description XML.') from error

    systems = root.findall('ros2_control')

    if not systems:
        raise RuntimeError('No ros2_control systems in robot description.')

    plugins = []

    for system in systems:
        plugin = system.findtext('hardware/plugin', '').strip()

        if not plugin:
            raise RuntimeError('A ros2_control system has no hardware plugin.')

        plugins.append(plugin)

    return plugins

def read_robot_description(node):
    import rclpy
    from rcl_interfaces.msg import ParameterType
    from rcl_interfaces.srv import GetParameters

    client = node.create_client(
        GetParameters,
        '/robot_state_publisher/get_parameters',
    )

    try:
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('Robot description parameter service unavailable.')

        request = GetParameters.Request()
        request.names = ['robot_description']

        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

        if not future.done():
            future.cancel()
            raise RuntimeError('Robot description request timed out.')

        response = future.result()

        if response is None or len(response.values) != 1:
            raise RuntimeError('Missing robot description response.')

        value = response.values[0]

        if value.type != ParameterType.PARAMETER_STRING:
            raise RuntimeError('Robot description must be a string.')

        return value.string_value

    finally:
        node.destroy_client(client)
        
def require_configured_hardware(node, *, expected_backend):
    expected_plugins = {
        'fake': 'uf_robot_hardware/UFRobotFakeSystemHardware',
        'real': 'uf_robot_hardware/UFRobotSystemHardware',
    }

    if expected_backend not in expected_plugins:
        raise ValueError('Expected backend must be real or fake.')

    plugins = read_hardware_plugins(node)
    expected_plugin = expected_plugins[expected_backend]

    if not plugins or any(plugin != expected_plugin for plugin in plugins):
        raise RuntimeError(
            f'Expected exclusively {expected_backend} hardware '
            f'in robot_description; found: {plugins}'
        )

    node.get_logger().info(
        f'Robot description matches the {expected_backend} backend.'
    )

    return plugins

def read_hardware_components(node):
    import rclpy
    from controller_manager_msgs.srv import ListHardwareComponents

    client = node.create_client(
        ListHardwareComponents,
        '/controller_manager/list_hardware_components',
    )

    try:
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError('Hardware component service unavailable.')

        future = client.call_async(ListHardwareComponents.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

        if not future.done():
            future.cancel()
            raise RuntimeError('Hardware component request timed out.')

        response = future.result()

        if response is None or not response.component:
            raise RuntimeError('No hardware components returned.')

        return response.component

    finally:
        node.destroy_client(client)
        
        
def require_hardware_environment(node, *, expected_backend):
    configured_plugins = require_configured_hardware(
        node,
        expected_backend=expected_backend,
    )

    components = read_hardware_components(node)
    loaded_plugins = []

    for component in components:
        plugin = (
            getattr(component, 'plugin_name', '')
            or getattr(component, 'class_type', '')
        )

        if component.state.label != 'active':
            raise RuntimeError(
                f'Hardware component is not active: '
                f'{component.name}; state={component.state.label}'
            )

        loaded_plugins.append(plugin)

    if sorted(loaded_plugins) != sorted(configured_plugins):
        raise RuntimeError(
            'Configured and loaded hardware plugins do not match. '
            f'Configured: {configured_plugins}; '
            f'loaded: {loaded_plugins}'
        )

    node.get_logger().info(
        f'Hardware environment verified: {expected_backend}; '
        f'{len(components)} active components.'
    )

    return components

def read_hardware_plugins(node):
    description = read_robot_description(node)
    return hardware_plugins_from_description(description)


def read_arm_joint_limits(node):
    try:
        root = ET.fromstring(read_robot_description(node))
    except ET.ParseError as error:
        raise RuntimeError('Invalid robot description XML.') from error

    limits = {}

    for index in range(1, 7):
        name = f'joint{index}'
        limit = root.find(f"./joint[@name='{name}']/limit")

        if limit is None:
            raise RuntimeError(f'Missing URDF joint limits: {name}')

        try:
            lower = float(limit.attrib['lower'])
            upper = float(limit.attrib['upper'])
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                f'Invalid URDF joint limits: {name}'
            ) from error

        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower >= upper
        ):
            raise RuntimeError(f'Invalid URDF joint limits: {name}')

        limits[name] = (lower, upper)

    return limits

def require_spline_interpolation(node):
    import rclpy
    from rcl_interfaces.msg import ParameterType
    from rcl_interfaces.srv import GetParameters

    client = node.create_client(
        GetParameters,
        '/lite6_traj_controller/get_parameters',
    )

    try:
        if not client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(
                'Trajectory controller parameter service unavailable.'
            )

        request = GetParameters.Request()
        request.names = ['interpolation_method']

        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            node,
            future,
            timeout_sec=5.0,
        )

        if not future.done():
            future.cancel()
            raise RuntimeError(
                'Controller interpolation parameter request timed out.'
            )

        response = future.result()

        if response is None or len(response.values) != 1:
            raise RuntimeError(
                'Missing controller interpolation parameter.'
            )

        value = response.values[0]

        if (
            value.type != ParameterType.PARAMETER_STRING
            or value.string_value != 'splines'
        ):
            raise RuntimeError(
                'Validated trajectories require '
                'controller interpolation_method=splines.'
            )

        node.get_logger().info(
            'Trajectory controller interpolation verified: splines.'
        )

    finally:
        node.destroy_client(client)