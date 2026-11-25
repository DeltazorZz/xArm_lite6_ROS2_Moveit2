from contextlib import contextmanager
from copy import deepcopy
import math

from moveit_msgs.msg import CollisionObject, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity

from big_chemistry_motion.cartesian_motion import _call_service
from big_chemistry_motion.motion_checks import is_tiny_support_contact


PLATE_IDS = frozenset(('plate_base', 'plate_body'))


def _read_scene(node):
    request = GetPlanningScene.Request()
    request.components.components = (
        PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
    )
    return _call_service(
        node, GetPlanningScene, '/get_planning_scene', request
    ).scene


def _apply_objects(node, objects):
    request = ApplyPlanningScene.Request()
    request.scene.is_diff = True
    request.scene.robot_state.is_diff = True
    request.scene.world.collision_objects = deepcopy(objects)
    response = _call_service(
        node, ApplyPlanningScene, '/apply_planning_scene', request
    )
    if not response.success:
        raise RuntimeError('MoveIt rejected the precheck world update.')


def _same_object_geometry(actual, expected):
    # World-frame transforms may introduce negligible floating-point changes.
    if actual.header.frame_id != expected.header.frame_id:
        return False
    for field in (
        'primitives', 'primitive_poses', 'meshes', 'mesh_poses',
        'planes', 'plane_poses', 'subframe_names', 'subframe_poses',
    ):
        if getattr(actual, field) != getattr(expected, field):
            return False
    for field in ('x', 'y', 'z'):
        if not math.isclose(
            getattr(actual.pose.position, field),
            getattr(expected.pose.position, field),
            rel_tol=0.0, abs_tol=1e-9,
        ):
            return False
    a = actual.pose.orientation
    b = expected.pose.orientation
    # q and -q represent the same rotation.
    return any(
        all(math.isclose(
            getattr(a, field), sign * getattr(b, field),
            rel_tol=0.0, abs_tol=1e-9,
        ) for field in ('x', 'y', 'z', 'w'))
        for sign in (1.0, -1.0)
    )


def check_loaded_pickup_state(node, state):
    request = GetStateValidity.Request()
    request.robot_state = deepcopy(state)
    request.group_name = 'lite6'
    response = _call_service(
        node, GetStateValidity, '/check_state_validity', request
    )
    if response.valid:
        node.get_logger().info('Loaded pickup start state is collision-free.')
        return

    for contact in response.contacts:
        node.get_logger().warning(
            f'LOADED START CONTACT: {contact.contact_body_1} <-> '
            f'{contact.contact_body_2}, depth={contact.depth:.9g} m'
        )
    if is_tiny_support_contact(response):
        node.get_logger().info(
            'Loaded start has only a <=1 micrometre plate/support contact. '
            'No ACM exception is applied to Cartesian planning; '
            'subsequent lift states must separate from the support.'
        )
        return
    raise RuntimeError('Loaded pickup start state is invalid. No lift planning.')


@contextmanager
def temporary_plate_world_removal(node, loaded_state):
    if node.get_parameter('motion_goal').value == 'pickup_fake':
        from big_chemistry_motion.fake_pickup import require_fake_environment
        require_fake_environment(node)
    elif (
        node.get_parameter('motion_goal').value != 'precheck'
        or node.get_parameter('execute').value
    ):
        raise RuntimeError('Temporary plate removal requires preview-only precheck.')

    attachments = loaded_state.attached_collision_objects
    if (
        len(attachments) != 2
        or {item.object.id for item in attachments} != PLATE_IDS
    ):
        raise RuntimeError('Loaded precheck must contain both attached plate parts.')
    for item in attachments:
        if (
            item.link_name != 'link_eef'
            or item.object.header.frame_id != 'link_eef'
            or not (item.object.primitives or item.object.meshes)
        ):
            raise RuntimeError(f'Incomplete hypothetical attachment: {item.object.id}')

    original_scene = _read_scene(node)
    if original_scene.robot_state.attached_collision_objects:
        raise RuntimeError('Precheck requires no live attachments; reset the fake scene.')

    original = {
        obj.id: deepcopy(obj)
        for obj in original_scene.world.collision_objects if obj.id in PLATE_IDS
    }
    if set(original) != PLATE_IDS:
        raise RuntimeError('Both original world plate objects are required.')

    removals = []
    for object_id, obj in original.items():
        removal = CollisionObject()
        removal.header.frame_id = obj.header.frame_id
        removal.id = object_id
        removal.operation = CollisionObject.REMOVE
        removals.append(removal)

    try:
        # The try includes apply: a timed-out request may already have taken effect.
        _apply_objects(node, removals)
        scene = _read_scene(node)
        if any(obj.id in PLATE_IDS for obj in scene.world.collision_objects):
            raise RuntimeError('Duplicate world plates remain; precheck blocked.')
        if scene.robot_state.attached_collision_objects:
            raise RuntimeError('Live attachments changed during precheck setup.')
        node.get_logger().info(
            'PRECHECK SCENE: duplicate world plates removed; '
            'both plate parts remain attached in the hypothetical request state. '
            'Live robot joints and live attachments were not changed.'
        )
        yield
    finally:
        try:
            _apply_objects(node, list(original.values()))
            restored = _read_scene(node)
            objects = {obj.id: obj for obj in restored.world.collision_objects}
            if restored.robot_state.attached_collision_objects:
                raise RuntimeError('Unexpected live attachments after precheck.')
            if not all(
                object_id in objects
                and _same_object_geometry(objects[object_id], obj)
                for object_id, obj in original.items()
            ):
                raise RuntimeError('Restored plate geometry does not match the snapshot.')
        except Exception as error:
            node.get_logger().error(
                'PRECHECK SCENE RESTORE FAILED. Stop and restart the fake stack '
                f'before further planning: {error}'
            )
            raise RuntimeError('Could not verify precheck scene restoration.') from error
        node.get_logger().info(
            'PRECHECK SCENE RESTORED: both original world plates verified; '
            'no live attachments.'
        )
