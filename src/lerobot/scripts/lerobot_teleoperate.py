# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.factory import make_teleop_robot_processors
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    robotiq,
    denso_windows,
    denso_deltapose,
    denso_deltapose_force
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    quest,
    bi_spacemouse,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    spacemouse,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    quest_haptics,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        _ = robot.send_action(robot_action_to_send)

        teleop.send_feedback(obs)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            # print("\n" + "-" * (display_len + 10))
            # print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # # Display the final robot action that was sent
            # for motor, value in robot_action_to_send.items():
            #     print(f"{motor:<{display_len}} | {value:>7.2f}")
            # move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_teleop_robot_processors(teleopConfig=cfg.teleop, robotConfig=cfg.robot)

    teleop.connect()
    robot.connect()

    # Connect to robot server
    while False:
        try:

            import socket, json
            SERVER_IP = "192.168.2.105"
            TASK_PORT = 12344
            RECV_BUFFER = 4096
            SOCKET_TIMEOUT = 1.0

            def send_init_flag(sock):
                """
                Send the initFlag=1 message as JSON with a trailing newline.
                """
                msg = {"initFlag": 1,
                    "doubleinitFlag": 0
                    }
                data = (json.dumps(msg) + "\n").encode("utf-8")
                sock.sendall(data)

            def send_doubleinit_flag(sock):
                """
                Send the doubleinitFlag=1 message as JSON with a trailing newline.
                """
                msg = {"initFlag": 0,
                    "doubleinitFlag": 1
                    }
                data = (json.dumps(msg) + "\n").encode("utf-8")
                sock.sendall(data)

            def send_clear_init_flag(sock):
                """
                Send the doubleinitFlag=1 message as JSON with a trailing newline.
                """
                msg = {"initFlag": 0,
                    "doubleinitFlag": 0
                    }
                data = (json.dumps(msg) + "\n").encode("utf-8")
                sock.sendall(data)

            def read_robot_init_flag(sock, buffer=b""):
                """
                Read from the socket until a full line is received.
                Parse JSON and return the value of "robot_init" (default None).
                Returns (flag, leftover_bytes).
                """
                try:
                    chunk = sock.recv(RECV_BUFFER)
                    if not chunk:
                        # Connection closed
                        return None, buffer
                except socket.timeout:
                    return None, buffer

                buffer += chunk
                if b"\n" not in buffer:
                    return None, buffer

                line, rest = buffer.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode("utf-8").strip())
                    return msg.get("robot_init", None), rest
                except json.JSONDecodeError:
                    return None, rest
            # 1) create socket and connect to server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(SOCKET_TIMEOUT)
            sock.connect((SERVER_IP, TASK_PORT))
            print(f"[CLIENT] Connected to {SERVER_IP}:{TASK_PORT}")

            # 2) send the first initFlag
            send_init_flag(sock)
            print("[CLIENT] Sent initFlag=1, waiting for robot to acknowledge...")

            # 3) loop: send initFlag periodically and wait for robot_init==1
            while True:
                flag, buffer = read_robot_init_flag(sock)
                if flag == 1:
                    print("[CLIENT] Received robot_init=1, start second handshake.")
                    send_doubleinit_flag(sock)
                    print("[CLIENT] Sent doubleinitFlag=1, waiting for robot to acknowledge...")

                    while True:
                        flag, buffer = read_robot_init_flag(sock)
                        if flag == 0:
                            print("[CLIENT] Got robot initFlag reset, start the teleoperation loop.")
                            send_clear_init_flag(sock)
                            time.sleep(0.1)
                            break
                        else:
                            send_doubleinit_flag(sock)
                            time.sleep(0.1)
                    break
                else:
                    send_init_flag(sock)
                    time.sleep(0.1)
            # 4) close and exit
            sock.close()
            break

        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"[CLIENT] Connection failed ({e}), retrying in 1s...")
            try:
                sock.close()
            except:
                pass
            time.sleep(1)

        except Exception as e:
            print(f"[CLIENT] Unexpected error: {e}")
            try:
                sock.close()
            except:
                pass

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_devices()
    teleoperate()


if __name__ == "__main__":
    main()
