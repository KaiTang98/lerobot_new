import time
from collections import deque
from collections.abc import Sequence
from threading import Lock
from copy import deepcopy
from cv_bridge_simple import image_to_numpy, compressed_image_to_numpy
import argparse

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
import torchvision.transforms as transforms 

from lerobot.envs.utils import preprocess_observation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.msg import CompressedImage, CompressedImage
from threading import Thread

import simpleaudio as sa
# This dataset data consists of 
# - top camera image (cropped to 300x225)
# - front camera image (640x480)
# - wrist camera image (640x480)
# - end-effector x, y position
# - action x, y position
# - stage: hanging a T-shirt on a hanger using dual arms
# - wrench x, y, z forces and torques

repo_id = "leledeyuan/hanging-tshirt"
fps = 30
observer_image_shape = (3, 480, 640)  # C, H, W
task_name = ["Moving two arms from initial position to find hanger left corner that inserts the garment left sleeve.",
             "Primarily moving the left arm to hang the left sleeve on the hanger left corner, while the right arm as an assistant to keep the sleeve open but not too tight.",
             "Primarily moving the right arm to insert the right sleeve on the hanger right corner, while the left arm as an assistant to keep the sleeve. Usually this step has a large contact force.",
             "Lifting the garment body slightly to the center of the hanger by moving both arms upward to avoid the garment slipping off the hanger.",
             "Releasing and regrasping the garment from shoulder to the sleeves, then aligning the garment center to the hanger center by moving both arms to the center position.",
             ]

class DatasetsRecorder(Node):
    def __init__(self):
        super().__init__("datasets_recorder")
        self.lock = Lock()
        self.episode_data = []
        self.current_episode = None
        self.recording = False

        parser = argparse.ArgumentParser(description='Pusht Datasets Recorder')
        parser.add_argument('--resume', help='Load from existing dataset)', action='store_true')
        args = parser.parse_args()

        resume = args.resume
        print(f"Resume: {resume}")

        self.current_pose_sub_l = self.create_subscription(
            PoseStamped,
            "/left_cartesian_compliance_controller/current_pose",
            self.current_pose_callback_l,
            10,
        )
        self.current_pose_l = PoseStamped()

        self.current_action_sub_l = self.create_subscription(
            PoseStamped,
            "/left_cartesian_compliance_controller/target_frame_monitor",
            self.current_action_callback_l,
            10,
        )
        self.current_action_l = PoseStamped()

        self.current_wrench_sub_l = self.create_subscription(
            WrenchStamped,
            "/left_cartesian_compliance_controller/current_wrench",
            self.current_wrench_callback_l,
            10,
        )
        self.current_wrench_l = WrenchStamped()

        self.current_pose_sub_r = self.create_subscription(
            PoseStamped,
            "/right_cartesian_compliance_controller/current_pose",
            self.current_pose_callback_r,
            10,
        )
        self.current_pose_r = PoseStamped()

        self.current_action_sub_r = self.create_subscription(
            PoseStamped,
            "/right_cartesian_compliance_controller/target_frame_monitor",
            self.current_action_callback_r,
            10,
        )
        self.current_action_r = PoseStamped()

        self.current_wrench_sub_r = self.create_subscription(
            WrenchStamped,
            "/right_cartesian_compliance_controller/current_wrench",
            self.current_wrench_callback_r,
            10,
        )
        self.current_wrench_r = WrenchStamped()

        self.current_gripper_sub_l = self.create_subscription(
            Float64MultiArray,
            "/L_gripper_forward_position_controller/commands",
            self.current_gripper_callback_l,
            10,
        )
        self.current_gripper_l = 0.0

        self.current_gripper_sub_r = self.create_subscription(
            Float64MultiArray,
            "/R_gripper_forward_position_controller/commands",
            self.current_gripper_callback_r,
            10,
        )
        self.current_gripper_r = 0.0            


        self.front_camera_sub = self.create_subscription(
            CompressedImage,
            "/camera/table/color/image_raw/compressed",
            self.front_camera_callback,
            10,
        )
        self.front_camera_img = None

        self.top_camera_sub = self.create_subscription(
            CompressedImage,
            "/camera/top/color/image_raw/compressed",
            self.top_camera_callback,
            10,
        )
        self.top_camera_img = None

        self.wrist_camera_sub_l = self.create_subscription(
            CompressedImage,
            "/camera/wrist_l/color/image_rect_raw/compressed",
            self.wrist_camera_callback_l,
            10,
        )
        self.wrist_camera_img_l = None

        self.wrist_camera_sub_r = self.create_subscription(
            CompressedImage,
            "/camera/wrist_r/color/image_rect_raw/compressed",
            self.wrist_camera_callback_r,
            10,
        )
        self.wrist_camera_img_r = None

        self.start_recording_srv = self.create_service(
            Trigger,
            "start_recording",
            self.start_recording_callback,
        )
        self.stop_recording_srv = self.create_service(
            SetBool,
            "stop_recording",
            self.stop_recording_callback,
        )

        self.stage_srv = self.create_service(
            Trigger,
            "stage_up",
            self.stage_up_callback,
        )
        self.phase_stage = 0.0

        self.image_sub_srv = self.create_service(
            SetBool,
            "image_subscribe",
            self.image_subscribe_callback,
        )
        self.image_sub_start = True

        self.episode_start = False
        self.episode_over = False
        self.episode_saved = False

        self.start_time = rclpy.clock.Clock().now()

        self.reward = 0.0
        self.next_done = False
        # Create datasets
        state_action_names = ["ee_x_l", "ee_y_l", "ee_z_l", "ee_qx_l", "ee_qy_l", "ee_qz_l", "ee_qw_l", 
                              "force_x_l", "force_y_l", "force_z_l", "torque_x_l", "torque_y_l", "torque_z_l",
                              "gripper_l",
                              "ee_x_r", "ee_y_r", "ee_z_r", "ee_qx_r", "ee_qy_r", "ee_qz_r", "ee_qw_r", 
                              "force_x_r", "force_y_r", "force_z_r", "torque_x_r", "torque_y_r", "torque_z_r",
                              "gripper_r",
                              "stage"]
        features = {
            "observation.state": {
            "dtype": "float32",
            "shape": (len(state_action_names),),
            "names": state_action_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(state_action_names),),
                "names": state_action_names,
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": observer_image_shape,
                "names": ["channels", "height", "width"],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": observer_image_shape,
                "names": ["channels", "height", "width"],
            },
            "observation.images.wrist_l": {
                "dtype": "video",
                "shape": observer_image_shape,
                "names": ["channels", "height", "width"],
            },
            "observation.images.wrist_r": {
                "dtype": "video",
                "shape": observer_image_shape,
                "names": ["channels", "height", "width"],
            },
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }

        if resume:
            self.get_logger().info("Resuming from existing dataset")
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
            )
            self.dataset.start_image_writer(
                num_processes=0,
                num_threads=4 * 3,  # 4 threads per camera
            )
            self.num_episode = self.dataset.num_episodes
            print(f"Resumed dataset with {self.num_episode} episodes")
        else:
            self.get_logger().info("Creating new dataset")

            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=None,
                use_videos=True,
                image_writer_threads= 4 * 3, # 4 threads per camera
                image_writer_processes= 0,
                features=features,
            )
            self.num_episode = 1

        self.callback_time_front = rclpy.clock.Clock().now()
        self.callback_time_wrist_l = rclpy.clock.Clock().now()
        self.callback_time_wrist_r = rclpy.clock.Clock().now()

        self.saving_sound = sa.WaveObject.from_wave_file("saving.wav")
        self.saved_sound = sa.WaveObject.from_wave_file("win_sound.wav")


        # Create a thread to run the main loop
        self.thread = Thread(target=self.run)
        self.thread.start()
    # End of __init__

    def current_pose_callback_l(self, msg: PoseStamped):
        self.current_pose_l = msg

    def current_action_callback_l(self, msg: PoseStamped):
        self.current_action_l = msg

    def current_wrench_callback_l(self, msg: WrenchStamped):
        self.current_wrench_l = msg

    def current_pose_callback_r(self, msg: PoseStamped):
        self.current_pose_r = msg

    def current_action_callback_r(self, msg: PoseStamped):
        self.current_action_r = msg
    
    def current_wrench_callback_r(self, msg: WrenchStamped):
        self.current_wrench_r = msg

    def current_gripper_callback_l(self, msg: Float64MultiArray):
        self.current_gripper_l = msg.data[0]
    
    def current_gripper_callback_r(self, msg: Float64MultiArray):
        self.current_gripper_r = msg.data[0]

    def front_camera_callback(self, msg: CompressedImage):
        # self.front_camera_img = image_to_numpy(msg)
        # self.get_logger().info("Hz: {:.2f}".format(1.0 / ((rclpy.clock.Clock().now() - self.callback_time_front).nanoseconds / 1e9)))
        # self.callback_time_front = rclpy.clock.Clock().now()

        self.front_camera_img = compressed_image_to_numpy(msg)

    def top_camera_callback(self, msg: CompressedImage):
        # self.top_camera_img = image_to_numpy(msg)
        self.top_camera_img = compressed_image_to_numpy(msg)
        
    def wrist_camera_callback_l(self, msg: CompressedImage):
        # self.wrist_camera_img_l = image_to_numpy(msg)
        self.wrist_camera_img_l = compressed_image_to_numpy(msg)
    
    def wrist_camera_callback_r(self, msg: CompressedImage):
        # self.wrist_camera_img_r = image_to_numpy(msg)
        self.wrist_camera_img_r = compressed_image_to_numpy(msg)


    def start_recording_callback(self, request: SetBool.Request, response: SetBool.Response):
        self.get_logger().info("Starting recording")
        self.episode_start = True
        response.success = True
        self.start_time = rclpy.clock.Clock().now()
        response.message = "Recording started"
        return response
    
    def stop_recording_callback(self, request: SetBool.Request, response: SetBool.Response):
        self.get_logger().info("Stopping recording")
        self.episode_over = True
        self.phase_stage = 0.0
        if request.data:
            self.get_logger().info("Saving episode")
            self.episode_saved = True
            response.success = True
            response.message = "Recording stopped and episode saved"
        else:
            self.get_logger().info("Not saving episode")
            self.episode_saved = False
            response.success = True
            response.message = "Recording stopped but episode not saved"
        return response
    
    def stage_up_callback(self, request: Trigger.Request, response: Trigger.Response):
        if self.phase_stage + 1.0 >= len(task_name):
            self.get_logger().info(f"Stage is already at maximum {self.phase_stage}, cannot increase further")
        else:
            self.phase_stage += 1.0
            self.get_logger().info(f"Task stage increased to {self.phase_stage}: {task_name[int(self.phase_stage)]}")
        response.success = True
        response.message = f"Stage increased to {self.phase_stage}"
        return response

    def image_subscribe_callback(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            self.get_logger().info("Starting image subscription")
            self.image_sub_start = True
        else:
            self.get_logger().info("Stopping image subscription")
            self.image_sub_start = False
        response.success = True
        response.message = "CompressedImage subscription updated"
        return response

    def run(self):
        successes = []
        next_done_count = 0
        cpu = torch.device("cpu")
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.ToTensor()
        resize_compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480,640), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        print(f"fps : {fps}")
        while rclpy.ok():
            start_loop_s = time.perf_counter()
            # 
            # get current time
            loop_start_time = rclpy.clock.Clock().now()
            if self.episode_start:
                # episode time limit 60s stop the episode and drop it
                if (rclpy.clock.Clock().now() - self.start_time).nanoseconds / 1e9 > 60.0:
                    self.start_time = rclpy.clock.Clock().now()
                    self.episode_over = True
                    self.episode_saved = False
                    self.get_logger().info(f"Episode {self.num_episode} reached time limit, discarding")

                # Observation
                obs_processed = {
                    "observation.state": torch.tensor([
                        # left arm
                        self.current_pose_l.pose.position.x,
                        self.current_pose_l.pose.position.y,
                        self.current_pose_l.pose.position.z,
                        self.current_pose_l.pose.orientation.x,
                        self.current_pose_l.pose.orientation.y,
                        self.current_pose_l.pose.orientation.z,
                        self.current_pose_l.pose.orientation.w,
                        self.current_wrench_l.wrench.force.x,
                        self.current_wrench_l.wrench.force.y,
                        self.current_wrench_l.wrench.force.z,
                        self.current_wrench_l.wrench.torque.x,
                        self.current_wrench_l.wrench.torque.y,
                        self.current_wrench_l.wrench.torque.z,
                        self.current_gripper_l,
                        # right arm
                        self.current_pose_r.pose.position.x,
                        self.current_pose_r.pose.position.y,
                        self.current_pose_r.pose.position.z,
                        self.current_pose_r.pose.orientation.x,
                        self.current_pose_r.pose.orientation.y,
                        self.current_pose_r.pose.orientation.z,
                        self.current_pose_r.pose.orientation.w,
                        self.current_wrench_r.wrench.force.x,
                        self.current_wrench_r.wrench.force.y,
                        self.current_wrench_r.wrench.force.z,
                        self.current_wrench_r.wrench.torque.x,
                        self.current_wrench_r.wrench.torque.y,
                        self.current_wrench_r.wrench.torque.z,
                        self.current_gripper_r,
                        # stage
                        self.phase_stage,
                    ], dtype=torch.float32).to(cpu),
                    "observation.images.front": transform(self.front_camera_img).to(cpu),
                    "observation.images.top": transform(self.top_camera_img).to(cpu),
                    "observation.images.wrist_l": transform(self.wrist_camera_img_l).to(cpu),
                    "observation.images.wrist_r": transform(self.wrist_camera_img_r).to(cpu),
                }
                action_processed = {
                    "action": torch.tensor([
                        # left arm
                        self.current_action_l.pose.position.x,
                        self.current_action_l.pose.position.y,
                        self.current_action_l.pose.position.z,
                        self.current_action_l.pose.orientation.x,
                        self.current_action_l.pose.orientation.y,
                        self.current_action_l.pose.orientation.z,
                        self.current_action_l.pose.orientation.w,
                        self.current_wrench_l.wrench.force.x,
                        self.current_wrench_l.wrench.force.y,
                        self.current_wrench_l.wrench.force.z,
                        self.current_wrench_l.wrench.torque.x,
                        self.current_wrench_l.wrench.torque.y,
                        self.current_wrench_l.wrench.torque.z,
                        self.current_gripper_l,
                        # right arm
                        self.current_action_r.pose.position.x,
                        self.current_action_r.pose.position.y,
                        self.current_action_r.pose.position.z,
                        self.current_action_r.pose.orientation.x,
                        self.current_action_r.pose.orientation.y,
                        self.current_action_r.pose.orientation.z,
                        self.current_action_r.pose.orientation.w,
                        self.current_wrench_r.wrench.force.x,
                        self.current_wrench_r.wrench.force.y,
                        self.current_wrench_r.wrench.force.z,
                        self.current_wrench_r.wrench.torque.x,
                        self.current_wrench_r.wrench.torque.y,
                        self.current_wrench_r.wrench.torque.z,
                        self.current_gripper_r,
                        # stage
                        self.phase_stage,
                    ], dtype=torch.float32).to(cpu),
                }

                frame = {**obs_processed, **action_processed}
            
                self.reward = 0.0
                self.next_done = False

                # Episode over
                if next_done_count == 1:
                    self.episode_start = False
                    self.next_done = True
                    next_done_count +=1

                if self.episode_over:
                    self.reward = 1.0
                    self.get_logger().info(f"Episode {self.num_episode} finished")
                    self.episode_over = False
                    next_done_count = 1
                

                frame["next.reward"] = torch.tensor([self.reward], dtype=torch.float32).to(cpu)
                frame["next.done"] = torch.tensor([self.next_done], dtype=torch.bool).to(cpu)
                frame["task"] = task_name[int(self.phase_stage)]

                # time before adding frame
                time_before = rclpy.clock.Clock().now()
                duration_before_add_s = (time_before - loop_start_time).nanoseconds / 1e6
                # self.get_logger().info(f"Time before adding frame: {duration_before_add_s:.4f} ms")
                self.dataset.add_frame(frame)
                # time after adding frame
                duration_after_add_s = (rclpy.clock.Clock().now() - time_before).nanoseconds / 1e6
                # self.get_logger().info(f"Time adding frame: {duration_after_add_s:.4f} ms")

            if next_done_count == 2:
                next_done_count = 0
                self.get_logger().info(f"Number of episodes recorded: {self.num_episode}")
                if self.episode_saved:
                    self.get_logger().info(f"Saving episode {self.num_episode}")
                    self.dataset.save_episode()
                    # print with colors green
                    print(f"\033[32mEpisode {self.num_episode} saved\033[0m")
                    self.saved_sound.play()
                    self.num_episode += 1
                else:
                    self.get_logger().info(f"Discarding episode {self.num_episode}")
                    self.dataset.clear_episode_buffer()
                    self.get_logger().info(f"Episode {self.num_episode} discarded")

            # spin once
            dt_s = time.perf_counter() - start_loop_s
            busy_wait(1 / fps - dt_s)
        
        self.dataset.finalize()
        print("Shutting down recorder")
        
def main():
    rclpy.init()
    datasets_recorder = DatasetsRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(datasets_recorder)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        datasets_recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
