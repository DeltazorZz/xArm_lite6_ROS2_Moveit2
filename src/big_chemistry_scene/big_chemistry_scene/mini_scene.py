import rclpy
from rclpy.node import Node

from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


PLANNING_FRAME = "link_base"


def create_box(
    object_id: str,
    x: float,
    y: float,
    z: float,
    size_x: float,
    size_y: float,
    size_z: float,
) -> CollisionObject:

    collision_object = CollisionObject()

    collision_object.header.frame_id = PLANNING_FRAME
    collision_object.id = object_id

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [
        size_x,
        size_y,
        size_z,
    ]

    pose = Pose()

    pose.position.x = x
    pose.position.y = y
    pose.position.z = z

    pose.orientation.w = 1.0

    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(pose)

    collision_object.operation = CollisionObject.ADD

    return collision_object


class MiniSceneLoader(Node):

    def __init__(self):
        super().__init__("mini_scene_loader")

        self.client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

        self.get_logger().info(
            "Waiting for MoveIt planning scene service..."
        )

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Still waiting for /apply_planning_scene..."
            )

        self.load_scene()

    def load_scene(self):

        scene = PlanningScene()

        scene.is_diff = True
        scene.robot_state.is_diff = True

        # -------------------------------------------------
        # MINIATURE "OPENTRONS OPENING"
        #
        # All dimensions are in METERS.
        #
        # Robot base:
        #
        #          +X
        #           ↑
        #
        #     [ LEFT ]   opening   [ RIGHT ]
        #
        #             robot
        #
        # -------------------------------------------------

        wall_x = 0.35

        opening_width = 0.22

        side_wall_width = 0.20
        side_wall_height = 0.30

        wall_thickness = 0.04

        y_offset = (
            opening_width / 2.0
            + side_wall_width / 2.0
        )

        # Left side of opening
        left_wall = create_box(
            object_id="opentrons_left_wall",
            x=wall_x,
            y=y_offset,
            z=side_wall_height / 2.0,
            size_x=wall_thickness,
            size_y=side_wall_width,
            size_z=side_wall_height,
        )

        # Right side of opening
        right_wall = create_box(
            object_id="opentrons_right_wall",
            x=wall_x,
            y=-y_offset,
            z=side_wall_height / 2.0,
            size_x=wall_thickness,
            size_y=side_wall_width,
            size_z=side_wall_height,
        )

        # Top of opening
        total_width = (
            side_wall_width * 2.0
            + opening_width
        )

        top_wall = create_box(
            object_id="opentrons_top_wall",
            x=wall_x,
            y=0.0,
            z=0.34,
            size_x=wall_thickness,
            size_y=total_width,
            size_z=0.08,
        )

        # Fake table.
        # Deliberately slightly below link_base,
        # so the robot base does not start in collision.
        table = create_box(
            object_id="table",
            x=0.25,
            y=0.0,
            z=-0.07,
            size_x=1.00,
            size_y=1.00,
            size_z=0.04,
        )

        scene.world.collision_objects = [
            left_wall,
            right_wall,
            top_wall,
            table,
        ]

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        if future.result() is not None and future.result().success:
            self.get_logger().info(
                "Mini Big Chemistry planning scene loaded successfully."
            )
        else:
            self.get_logger().error(
                "Failed to apply planning scene."
            )


def main(args=None):

    rclpy.init(args=args)

    node = MiniSceneLoader()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
