import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from xarm_msgs.srv import PlanPose, PlanExec


# -------------------------------------------------
# TARGET
# -------------------------------------------------

TARGET_X = 0.50
TARGET_Y = 0.00
TARGET_Z = 0.18

# Roll = 180 deg, pitch = 0, yaw = 0
# Quaternion equivalent:
TARGET_QX = 1.0
TARGET_QY = 0.0
TARGET_QZ = 0.0
TARGET_QW = 0.0

# First keep this FALSE.
# We only want MoveIt to PLAN.
EXECUTE_AFTER_PLAN = False


class PosePlanner(Node):

    def __init__(self):
        super().__init__("big_chemistry_pose_planner")

        self.plan_client = self.create_client(
            PlanPose,
            "/xarm_pose_plan",
        )

        self.exec_client = self.create_client(
            PlanExec,
            "/xarm_exec_plan",
        )

    def wait_for_services(self):

        self.get_logger().info(
            "Waiting for xArm planner services..."
        )

        while not self.plan_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "Waiting for /xarm_pose_plan..."
            )

        while not self.exec_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "Waiting for /xarm_exec_plan..."
            )

        self.get_logger().info(
            "Planner services are ready."
        )

    def plan(self):

        target = Pose()

        target.position.x = TARGET_X
        target.position.y = TARGET_Y
        target.position.z = TARGET_Z

        target.orientation.x = TARGET_QX
        target.orientation.y = TARGET_QY
        target.orientation.z = TARGET_QZ
        target.orientation.w = TARGET_QW

        request = PlanPose.Request()
        request.target = target

        self.get_logger().info(
            "Requesting automatic trajectory planning..."
        )

        self.get_logger().info(
            f"Target XYZ: "
            f"{TARGET_X:.3f}, "
            f"{TARGET_Y:.3f}, "
            f"{TARGET_Z:.3f}"
        )

        future = self.plan_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            self.get_logger().error(
                "No response from planner."
            )
            return False

        if not response.success:
            self.get_logger().error(
                "MoveIt could not find a valid "
                "collision-free trajectory."
            )
            return False

        self.get_logger().info(
            "SUCCESS: collision-free trajectory found."
        )

        return True

    def execute(self):

        request = PlanExec.Request()
        request.wait = True

        self.get_logger().info(
            "Executing planned trajectory..."
        )

        future = self.exec_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is not None and response.success:
            self.get_logger().info(
                "Trajectory executed successfully."
            )
        else:
            self.get_logger().error(
                "Trajectory execution failed."
            )


def main(args=None):

    rclpy.init(args=args)

    node = PosePlanner()

    node.wait_for_services()

    success = node.plan()

    if success and EXECUTE_AFTER_PLAN:
        node.execute()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()

