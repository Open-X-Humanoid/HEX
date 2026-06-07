import os
import cv2
import time
import torch
import threading
import numpy as np
from PIL import Image
from pynput import keyboard
from collections import deque

from hex.model.tools import read_mode_config
from hex.model.framework.base_framework import baseframework
from hex.dataloader.gr00t_lerobot.transform.state_action import Normalizer

from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.utils.logger.logger_loader import logger
from xrocs.core.station_loader import StationLoader

preparing = True
robot_stand = True


def actions_interpolation(pre_action, actions, target_idx=10, chunk_size=20):
    target_action = actions[target_idx].copy()
    result = [pre_action]
    inter_path = np.linspace(pre_action, target_action, chunk_size + 2)
    result.extend(inter_path[1:])
    
    result = np.array(result)
    for j in [7, 15, 26]:
        result[:, j] = (result[:, j] > 0.5).astype(float)
    
    if len(result) > chunk_size + 1:
        result = result[:chunk_size + 1]
    else:
        while len(result) < chunk_size + 1:
            result = np.append(result, [result[-1][np.newaxis, :]], axis=0)
            
    final_actions = result[1:chunk_size + 1]
    return final_actions


def on_press(key):
    global preparing
    try:
        if key == keyboard.Key.enter:
            preparing = False

    except AttributeError:
        pass


def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


class DataInference:
    def __init__(self, ):
        config_path = "/home/mig/Documents/configuration.toml"
        cfg_loader = ConfigLoader(config_path)
        cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        print("initial done")
        
        # model
        model_path = './pretrained_models/hex/EAI_real_world_imitate_gesture_2B/hex_ac100_2w_8gpu_state_query_history2_ft/checkpoints/steps_20000_pytorch_model.pt'
        self.infer_model = baseframework.from_pretrained(model_path)
        use_bf16 = True
        if use_bf16: 
            self.infer_model = self.infer_model.to(torch.bfloat16)
        self.infer_model = self.infer_model.to("cuda").eval()

        model_config, norm_stats = read_mode_config(model_path)
        unnorm_key = None
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        action_norm_stats = norm_stats[unnorm_key]["action"]
        state_norm_stats = norm_stats[unnorm_key]["state"]
        self.normalizer_state = Normalizer('q99', state_norm_stats)
        self.normalizer_action = Normalizer('q99', action_norm_stats)

        self.instruction = ["Throw the paper in the trash and put the rest in the box"]
        self.tags = ['tienkung3_v4']

        self.num_camera = 1
        self.need_smooth = False

        self.switch_to_stand_mode = False
        self.history_mode = 0
        self.count_switch = 0

    def reset_to_home(self,):
        # arm
        both_home = Joints([2.1724987e-01,  4.8466492e-01,  1.2493324e-01, -1.9220648e+00, 5.3654575e-01, -1.7166138e-03, -1.7681408e-01,
                            2.3785019e-01, -4.7245789e-01, -2.1229076e-01, -1.8278399e+00, -7.2232437e-01,  8.4877968e-02,  1.3294411e-01], num_of_dofs=14)
        self.robot_station.get_robot_handle()['robot'].reach_target_joint(both_home)

        # head
        # self.robot_station.get_robot_handle()['robot'].set_head_joints_position([-0.0396719, 0.12336826, 0.07951498])

        # hand
        self.robot_station.get_gripper_handle()['left'].open()
        self.robot_station.get_gripper_handle()['right'].open()
        time.sleep(1)
        print("reset done")

    @torch.inference_mode()
    def inference(self):
        listener_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
        listener_thread.start()

        global preparing
        global robot_stand

        print("Going to start position")
        print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
        os.system("espeak start")

        print("enter Enter to go")
        global preparing
        while preparing:
            self.robot_station.get_obs()
            time.sleep(0.5)
        preparing = True

        obs = self.robot_station.get_obs() 
        obs = self.robot_station.get_obs()
        print(f'get obs: {obs}')
        cam_img = obs['images']['camera']
        cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)  # convert to RGB    [480, 640, 3]
        cam_img = Image.fromarray(cam_img).resize((224, 224))  # numpy -> PIL
        
        image_queue = deque(maxlen=self.num_camera)
        image_queue.append(cam_img)
        step = 0
        while robot_stand:
            cam_img = obs['images']['camera']
            cam_img = cv2.imdecode(cam_img, cv2.IMREAD_COLOR)
            cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)  # convert to RGB    [480, 640, 3]
            cam_img = Image.fromarray(cam_img).resize((224, 224))  # numpy -> PIL
            image_queue.append(cam_img)
            batch_images = [list(image_queue)]

            state = np.concatenate([  
                obs['arm_joints']['left'][None, ...],   # 7 dims  
                obs['hand_joints']['left'][None, ...],  # 6 dims  
                obs['hand_touch']['left'][None, ...], # 20 dims  
                obs['arm_joints']['right'][None, ...],  # 7 dims  
                obs['hand_joints']['right'][None, ...], # 6 dims  
                obs['hand_touch']['right'][None, ...], # 20 dims  
                obs['waist_joints'][None, ...],        # 3 dims  
                obs['head_joints'][None, ...],        # 3 dims  
                obs['leg_move_velocity'][None, ...],    # 6 dims
                obs['head_imu_data'][None, ...],  # 10 dims  
                obs['leg_joints']['left'][None, ...], # 6 dims  
                obs['leg_joints']['right'][None, ...], # 6 dims  
            ], axis=-1) 
            state = torch.from_numpy(state)
            state = self.normalizer_state.forward(state).unsqueeze(0)

            action_pred: np.ndarray = self.infer_model.predict_action(batch_images, self.instruction, state, self.tags)['normalized_actions'][0]
            raw_actions = self.normalizer_action.inverse(torch.tensor(action_pred))
            raw_actions = np.array(raw_actions)
            raw_actions[:, 27] = raw_actions[:, 21]

            if self.need_smooth:
                if step == 0:
                    pre_action = raw_actions[0]
                raw_actions = actions_interpolation(pre_action, raw_actions, 20, chunk_size=15)
            else:
                raw_actions = raw_actions[:20]

            for raw_action in raw_actions:
                raw_action[7] = np.clip(raw_action[7], 0, 1)
                raw_action[15] = np.clip(raw_action[15], 0, 1)

                action_dict = {
                    "arm": {
                        "position": {
                            "left": np.asarray(raw_action[0:7]),
                            "right": np.asarray(raw_action[8:15]),
                        }
                    },
                   "hand": {
                        "position": {
                            "left": np.asarray([1]) if np.asarray([raw_action[7]]) > 0.5 else np.asarray([0]),
                            "right": np.asarray([1]) if np.asarray([raw_action[15]]) > 0.5 else np.asarray([0]),
                        }
                    },
                    "leg": {"float_base": raw_action[22:]}, 
                }
                # print('action_dict', action_dict)

                input(f"press to go")
                self.robot_station.execute_action(action_dict)
                obs = self.robot_station.get_obs()
                time.sleep(0.01)
            
            pre_action = raw_action
            step += 1

    def _check_unnorm_key(self, norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}")
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}")
        return unnorm_key


if __name__ == '__main__':
    data = DataInference()
    data.reset_to_home()
    data.inference()
