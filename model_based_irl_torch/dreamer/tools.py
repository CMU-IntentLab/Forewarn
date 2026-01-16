import collections
import io
import json
import os
import pathlib
import random
import time
import cv2
from tqdm import trange
from collections import defaultdict
from typing import Any, Callable, Union
from gym.spaces import Dict, Box, Discrete
import h5py
import numpy as np
import torch
from termcolor import cprint
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.utils import get_dataset_path_and_meta_info, get_robocasa_dataset_path_and_env_meta, get_real_dataset_path_and_env_meta, get_real_classifier_dataset_path_and_env_meta
import dreamer.networks as networks
import pickle
import wandb
def to_np(x):
    return x.detach().cpu().numpy()


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
    def __init__(self, model, always_frozen_layers=[]):
        self._model = model
        self._always_frozen_layers = always_frozen_layers

    def __enter__(self):
        for name, param in self._model.named_parameters():
            if name in self._always_frozen_layers:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class DebugLogger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

        # Initialize WandB

    def config(self, config_dict):
        pass
        
    def scalar(self, name, value):
        pass
        
    def image(self, name, value):
        pass
    def video(self, name, value):
        pass
    def write(self, fps=False, step=False, fps_namespace="", print_cli=True):
        pass
        
    def _compute_fps(self, step):
        pass

    def offline_scalar(self, name, value, step):
        pass

    def offline_video(self, name, value, step):
        pass


class Logger:
    def __init__(self, logdir, step , name):
        self._logdir = logdir
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

        name = name + '_' + str(logdir).split('/')[-2] + '_' + str(logdir).split('/')[-1]
        # Initialize WandB
        wandb.init(project="failure_prediction", config={"logdir": str(logdir)}, name=name)

    def config(self, config_dict):
        # Convert PosixPath objects to strings
        config_dict = {
            k: str(v) if isinstance(v, pathlib.PosixPath) else v
            for k, v in config_dict.items()
        }
        # Log the config
        wandb.config.update(config_dict)

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False, fps_namespace="", print_cli=True):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            fps_str = fps_namespace + "fps"
            scalars.append((fps_str, self._compute_fps(step)))
        if print_cli:
            print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        # Log metrics to WandB
        metrics = {"step": step, **dict(scalars)}
        wandb.log(metrics, step=step)
        
        for name, value in self._images.items():
            # Log images to WandB
            if np.shape(value)[0] == 3:
                value = np.transpose(value, (1, 2, 0))
            wandb.log({name: [wandb.Image(value, caption=name)]}, step=step)

        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            # Log videos to WandB
            wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)

        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        # Log scalar metrics to WandB
        wandb.log({f"scalars/{name}": value}, step=step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        # Log videos to WandB
        wandb.log({name: wandb.Video(value, fps=16, format="mp4")}, step=step)

def merge_two_dicts_no_relabel(cache1, cache2):
    """
    Merges two cache dictionaries without relabeling the trajectories.
    
    Args:
        cache1 (dict): The main cache dictionary.
        cache2 (dict): The cache dictionary to be merged into cache1.
        
    Returns:
        cache1 (dict): Updated cache1 containing both the original and new trajectories.
    """
    

    
    # Add trajectories from cache2 to cache1, with failure label (0)
    last_key = int(list(cache1.keys())[-1].split('_')[-1])
    for i, key in enumerate(cache2.keys()):
        new_key = f"exp_traj_{i + last_key+1}"
        cache1[new_key] = cache2[key].copy()  # Copy the trajectory to avoid mutation
    
    return cache1

def merge_two_cache_dicts(cache1, cache2):
    """
    Merges two cache dictionaries and assigns labels for success (1) and failure (0).
    
    Args:
        cache1 (dict): The main cache dictionary with trajectories labeled as success (1).
        cache2 (dict): The cache dictionary to be merged into cache1 with trajectories labeled as failure (0).
        
    Returns:
        cache1 (dict): Updated cache1 containing both the original and new trajectories, with appropriate labels.
    """
    

    # Add labels for success (1) in cache1
    for key in cache1.keys():
        ## create 1 label as the same length of the other elements
        
        cache1[key]['label'] = np.ones( cache1[key]['is_first'].shape)# Label success as 1
    
    # Add trajectories from cache2 to cache1, with failure label (0)
    last_key = int(list(cache1.keys())[-1].split('_')[-1])
    for i, key in enumerate(cache2.keys()):
        new_key = f"exp_traj_{i + last_key+1}"
        cache1[new_key] = cache2[key].copy()  # Copy the trajectory to avoid mutation
        cache1[new_key]['label'] = np.zeros( cache2[key]['is_first'].shape)  # Label failure as 0
    
    return cache1
        


def fill_expert_dataset_real_data(config, cache, is_val_set=False, padding=None):
    env_name = config.task.split("_", 1)[0]
    sample_freq = config.sample_freq
    selected_obs_keys = config.obs_keys
  
    env_names = [env_name]

    # Initialize extra info to return
    norm_dict = None
    state_dim = None
    action_dim = None
    total_id = 0

    for env_name_id, env_name in enumerate(env_names):
        ## load the dataset path and the env meta
       
        dataset_path, _ = get_real_dataset_path_and_env_meta(
            env_id=env_name,
            config = config,
            done_mode=config.done_mode,
        )
       
        f = h5py.File(dataset_path, "r")
        ## get the dir of dataset_path
        dataset_dir = os.path.dirname(dataset_path)
        # read the norm_dict
        if config.action_type == 'joint':
            with open(os.path.join(dataset_dir, f'norm_dict_joint_position.json'), 'r') as file:
                norm_dict = json.load(file)
        else:
            with open(os.path.join(dataset_dir, f'norm_dict_{config.action_type}.json'), 'r') as file:
                norm_dict = json.load(file)
        ## make the values of the norm_dict to be np.array
        for key in norm_dict.keys():
            norm_dict[key] = np.array(norm_dict[key])
        
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        # demos = [demos[i] for i in inds]
        ## load ids from the file ids.npy
        ids = np.load(os.path.join(dataset_dir, f'ids.npy'))
        demos = [demos[i-1] for i in ids]
      
        
        # if is_val_set, we don't fill the first num_exp_trajs which are used for training
        config.num_exp_trajs = (
            len(demos) if config.num_exp_trajs == -1 else config.num_exp_trajs
        )
        if is_val_set:
            
            if config.num_exp_trajs >= len(demos):
                print('not enough expert data for val')
                burn_in_trajs = 0 if is_val_set else 0 
            else:
                burn_in_trajs = config.num_exp_trajs if is_val_set else 0
        else: 
            burn_in_trajs = config.num_exp_trajs if is_val_set else 0
        num_fill_trajs = (
            min(len(demos), config.num_exp_trajs + config.validation_mse_trajs)
            if is_val_set
            else config.num_exp_trajs
        )
       
        obs_keys = f['data'][demos[0]]['obs'].keys()    
        pixel_keys = sorted([key for key in obs_keys if "image" in key and key in selected_obs_keys])
       
        state_keys = config.state_keys
        
        # Initialize norm_dict if it is None
        if norm_dict is None:
            # Read ob_dim and ac_dim from the first datapoint in the first demo
            first_demo = f["data"][demos[0]]
            ob_dim = 0
            for key in state_keys:
                ob_dim += np.prod(first_demo["obs"][key].shape[1:])
            ac_dim = first_demo["actions"].shape[1]

            print(f"Initizalizing norm_dict with ob_dim={ob_dim} and ac_dim={ac_dim}")
            norm_dict = {
                "ob_max": -np.inf * np.ones(ob_dim, dtype=np.float32),
                "ob_min": np.inf * np.ones(ob_dim, dtype=np.float32),
                "ac_max": -np.inf * np.ones(ac_dim, dtype=np.float32),
                "ac_min": np.inf * np.ones(ac_dim, dtype=np.float32),
            }
        origin_shape = list(f["data"][demos[0]]["actions_abs"].shape[1:])
        
        action_space = Box(-1, 1, shape = tuple(origin_shape))
        first_demo = f["data"][demos[0]]
       
        # Set state_dim and action_dim
        if state_dim is None:
            state_dim = 0
            for key in state_keys:
                state_dim += np.prod(first_demo["obs"][key].shape[1:])

        if action_dim is None:
            ## remove the last base + mode action because it is not used in the policy 
            ## pos: 3dim, axis_angle: 3dim, gripper: 1dim, base: 4 dim, mode: 1 dim
            
            action_dim = first_demo["actions_abs"].shape[1]             
        obs_space = {}
        if config.action_type == 'delta':
            action_key = "actions"
        elif config.action_type == 'abs':
            action_key = "actions_abs"
        elif config.action_type == 'joint':
            action_key = "actions_joint_position"
        else: 
            raise ValueError('action_type should be either delta or abs')
        
        for key in pixel_keys:
            obs_space[key] = Box(0, 1, shape = f["data"][demos[0]]["obs"][key].shape[1:])
        for key in state_keys:
            obs_space[key] = Box(-1, 1, shape = f["data"][demos[0]]["obs"][key].shape[1:])
        obs_space['is_terminal'] = Discrete(2)
        obs_space['is_first'] = Discrete(2)
        obs_space['is_last'] = Discrete(2)
        obs_space['discount'] = Box(0, 1, shape = (1,))
      
        obs_space['state'] = Box(-1, 1, shape = (state_dim,))
        
        observation_space = Dict(obs_space)
        
        for i, demo in tqdm(
            enumerate(demos),
            desc="Loading in expert data",
            ncols=0,
            leave=False,
            total=num_fill_trajs,
        ):
            if i < burn_in_trajs:
                continue
            elif i >= num_fill_trajs:
                print('last demo index', demos[i])
                break

            traj = f["data"][demo]
            label = f["data"][demo].attrs['label']
            # Concat state keys to create "state" key
            concat_state = []
            for t in range(len(traj["obs"][pixel_keys[0]])):
                curr_obs_state_vec = [traj["obs"][obs_key][t] for obs_key in state_keys]
                curr_obs_state_vec = np.concatenate(
                    curr_obs_state_vec, dtype=np.float32
                )
                concat_state.append(curr_obs_state_vec)

                
            # Stack Observations for State and Pixel Keys
            stacked_obs = {}
       

            stacked_obs["state"] = concat_state
            for key in pixel_keys:
                stacked_obs[key] = traj["obs"][key]
           
            length = len(traj["obs"][pixel_keys[0]])
        
            
              
            if length == 160 and env_name == 'GraspCup':
                ## for real data for cup, we only use part of the rollout after the first 41 steps
                ## for demonstrations, we use the entire rollout
                ind_step = 41
                
            else: 
                if hasattr(config, "classifier_filter") and config.classifier_filter == 'no_demo':
                    continue
                ind_step = 0
            cache[f'exp_traj_{total_id}_{ind_step}']  = {}
            for key in pixel_keys:
                obs_from_ind = np.array(traj["obs"][key])[ind_step::sample_freq]
                ## concatenate the obs[key][1] to the beginning of the list 
                if length == 160 and env_name == 'GraspCup':
                    obs_from_ind = np.concatenate([np.array(traj["obs"][key][1:2]), obs_from_ind], axis=0)
                    
                cache[f'exp_traj_{total_id}_{ind_step}'][key] = obs_from_ind
            state_from_ind = stacked_obs["state"][ind_step::sample_freq]
            if length == 160 and env_name == 'GraspCup':
                state_from_ind = np.concatenate([np.array(stacked_obs["state"][1:2]), state_from_ind], axis=0)
            ## normalize the state
            normalized_state = (state_from_ind - norm_dict["ob_min"]) / (norm_dict["ob_max"] - norm_dict["ob_min"])
            normalized_state = 2 * normalized_state - 1
            cache[f'exp_traj_{total_id}_{ind_step}']['state'] = normalized_state 
            subsample_actions = np.array(traj[action_key])[sample_freq -1+ind_step::sample_freq]
            if (length+ind_step) % sample_freq != 0:
                ## add the last action to the end of the list
                subsample_actions = np.concatenate([subsample_actions, np.array(traj[action_key][-1:])], axis=0)
            if length == 160 and env_name == 'GraspCup':
                subsample_actions = np.concatenate([np.array(traj[action_key][ind_step-1:ind_step]), subsample_actions], axis=0)
            ## normalize the actions based on ac_min, ac_max
            normalized_actions = (subsample_actions - norm_dict["ac_min"]) / (norm_dict["ac_max"] - norm_dict["ac_min"])
            normalized_actions = 2 * normalized_actions - 1
            cache[f'exp_traj_{total_id}_{ind_step}']['action'] = normalized_actions #subsample_actions
            size  = subsample_actions.shape[0]
            cache[f'exp_traj_{total_id}_{ind_step}']['discount'] = np.array([1]*size, dtype=np.float32)
            cache[f'exp_traj_{total_id}_{ind_step}']['is_last'] = np.array([0] * size, dtype=np.bool_)
            cache[f'exp_traj_{total_id}_{ind_step}']['is_terminal'] = np.array([0] * size, dtype=np.bool_)
            cache[f'exp_traj_{total_id}_{ind_step}']['is_first'] = np.array([1] + [0]*(size-1), dtype=np.bool_)
            ## check if all the keys have the same length
            for key in cache[f'exp_traj_{total_id}_{ind_step}']:
                if len(cache[f'exp_traj_{total_id}_{ind_step}'][key]) != size:
                    wrong_length = len(cache[f'exp_traj_{total_id}_{ind_step}'][key])
                    print(f'key {key} has length {wrong_length} but should have length {size}')

                    raise ValueError(f'key {key} has length {wrong_length} but should have length {size}')
            
            
            total_id += 1
           
        
        if not is_val_set:
            cprint(
                f"Loading expert buffer with {config.num_exp_trajs} trajectories from {dataset_path}",
                color="magenta",
                attrs=["bold"],
            )
        else:
            cprint(
                f"Loading validation buffer with {config.validation_mse_trajs} trajectories from {dataset_path}",
                color="magenta",
                attrs=["bold"],
            )
    f.close()
    return  observation_space, action_space, norm_dict, state_dim, action_dim










def evaluate_mse_trajectories(agent_eval, expert_eps, final_config):
    keys = list(expert_eps.keys())  # Convert keys to a list
    total_mse = 0
    for i in range(final_config.validation_mse_trajs):
        data_traj_i = expert_eps[keys[i]]

        # convert data_traj_i from dict of lists to dict of np arrays (stacked)
        data_traj_i_keys = list(data_traj_i.keys())
        for key in data_traj_i_keys:
            data_traj_i[key] = np.stack(data_traj_i[key])

        # get mse for trajectory i
        traj_mse = get_agent_mse_batchlen(agent_eval, data_traj_i)
        total_mse += traj_mse

    total_mse /= final_config.validation_mse_trajs
    return total_mse


def get_agent_mse_batchlen(agent_eval, data_traj_i):
    # data_traj_i is a dict of np arrays, each with shape (len_traj, ...)
    len_traj = len(data_traj_i["action"])
    avg_mse = 0
    agent_eval.reset()
    for j in range(len_traj):
        data_j = {k: v[j] for k, v in data_traj_i.items()}
        if data_j["is_first"]:
            agent_eval.reset()
        action = agent_eval.get_action(data_j)
        true_action = data_j["action"]
        mse = np.mean((action["action"] - true_action) ** 2)
        avg_mse += mse
    avg_mse /= len_traj
    return avg_mse




def add_to_cache(cache, _id, transition):
    if _id not in cache:
        cache[_id] = dict()
        for key, val in transition.items():
            cache[_id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[_id]:
                # fill missing data(action, etc.) at second time
                cache[_id][key] = [convert(0 * val)]
                cache[_id][key].append(convert(val))
            else:
                cache[_id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(generator))
            except StopIteration:
                raise StopIteration
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data
        
def from_eval_generator(generator, batch_size):
    """
    Takes a generator and yields batches of data until the generator is exhausted.
    """
    while True:
        batch = []
 
        for _ in range(batch_size):
            try:
                new_batch = next(generator)
            except RuntimeError:
                
                # If the generator is exhausted, return the partial batch if it exists
                if batch:
                    data = {}
                    for key in batch[0].keys():
                        data[key] = []
                        for i in range(len(batch)):
                            data[key].append(batch[i][key])
                        data[key] = np.stack(data[key], 0)
                    yield data
                return  # Stop when the generator is fully exhausted
            batch.append(new_batch)
        
        # Organize the batch into data dictionary (similar structure as single sample)
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)

        yield data       
# def from_traj_generator(generator, batch_size):
class EpisodeSampler:
    def __init__(self, episodes, length, seed=0):
        self.episodes = episodes
        self.length = length
        self.episode_keys = list(episodes.keys())  # List of episode keys
        self.episode_positions = {key: 0 for key in self.episode_keys}  # Position within each episode
        self.episode_idx = 0  # Current episode index
        self.np_random = np.random.RandomState(seed)
 

    def sample_episodes(self):
        """
        Iteratively sample through all episodes without resetting until the end.
        """
        while self.episode_idx < len(self.episode_keys):
            episode_key = self.episode_keys[self.episode_idx]
            episode = self.episodes[episode_key]
            total = len(next(iter(episode.values())))

            if total < 2:
                self.episode_idx += 1  # Skip empty or small episodes
                continue

            start_index = self.episode_positions[episode_key]
            end_index = min(start_index + self.length, total)
            # Collect transitions from the current episode
            ret = {k: v[start_index:end_index].copy() for k, v in episode.items() if "log_" not in k}

            if "is_first" in ret:
                ret["is_first"][0] = True

            # Update the position in the current episode
            self.episode_positions[episode_key] = end_index # start_index +1 #

            # If the current episode is exhausted, move to the next episode
            if end_index + self.length >= total:
            # if start_index + 1 + self.length >= total:
                self.episode_idx += 1

            yield ret

        # When all episodes are exhausted, stop iteration
        raise StopIteration("All episodes have been sampled.")

class EpisodeSpecialSampler(EpisodeSampler):
    def __init__(self, episodes, length, seed=0):
        
        self.episodes = episodes
        self.length = length
        self.episode_keys = ['exp_traj_100','exp_traj_101','exp_traj_102', 'exp_traj_103','exp_traj_104', 'exp_traj_105', 'exp_traj_106', 'exp_traj_107','exp_traj_108',  'exp_traj_109'] #',#'exp_traj_106']  #'exp_traj_101',  #'exp_traj_104'] #list(episodes.keys())  # List of episode keys
        self.episode_positions = dict(exp_traj_100 = 800, exp_traj_101 = 1400, exp_traj_102=880, exp_traj_103=420, exp_traj_104= 700,exp_traj_105 = 600,exp_traj_106=1350, exp_traj_107 = 420, exp_traj_108 = 900,exp_traj_109=1400)#exp_traj_104 =900 ) #{key: 880 for key in self.episode_keys}#{key: 0 for key in self.episode_keys}  # Position within each episode
        self.episode_idx =0 # 0  # Current episode index
        self.np_random = np.random.RandomState(seed)
        # for key in self.episode_keys:
        #     episode = self.episodes[key]
        #     ## save the trajectory as a video
        #     img_sequence = episode["robot0_agentview_front_image"]
        #     img_with_text = []
        #     for num,img in enumerate(img_sequence):
        #         image_height, image_width, _ = img.shape
    
        #         # Define font, scale, color, and thickness
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 0.4
        #         font_color = (255, 255, 255)  # White color
        #         font_thickness = 1
        #         text = f"Frame: {num}"
        #         # Calculate text size and position (centered at the bottom)
        #         text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        #         text_x = (image_width - text_size[0]) // 2
        #         text_y = image_height - 10  # 10 pixels from the bottom
        #         frame_with_text = cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        #         img_with_text.append(frame_with_text)
        #     ## use imageio to save the video
        #     imageio.mimsave(f'./failure_{key}.gif', img_with_text, fps=30)
class EvalEpisodeSampler(EpisodeSampler):
    def __init__(self):
        self.episode_id = 0
    def sample_partial_episodes(self, episodes, length, seed=0, max_length=99, mode='last', start_index = 0):
        """
        This function samples part of each episodes from a given dataset of episodes. It ensures the `is_first`
        flag is correctly set for each episode.

        Args:
            episodes (dict): Dictionary containing multiple episodes. Each episode is a dictionary
                            with keys for observations, actions, etc.
            seed (int): Random seed for sampling episodes.

        Yields:
            dict: A dictionary containing the sampled full episode data, with `is_first` handled properly.
        """
        # np_random = np.random.RandomState(seed)
        
        while True:
            # Randomly select an episode from the dataset
            # episode_key = np_random.choice(list(episodes.keys()))
            try :
                episode_key = list(episodes.keys())[self.episode_id]
                self.episode_id += 1
            except:
                raise StopIteration
            episode = episodes[episode_key]
            ## save actual length
            actual_length = len(episode["is_first"].copy())
            # print('actual length', actual_length)
            
            ## The original size is (B, T, D), now we need to shape into (B*T/16, 16, D)
            ## get the folded size of the episode
            # Copy the full episode data
            # ret = {k: v.copy() for k, v in episode.items() if "log_" not in k}
            ## copy the full episode data and pad the item to max_length, item can be obs, action, etc.
            ## item can be 1dim array, 2 dim array, 3 dim array, etc.
            if actual_length > max_length:
                actual_length = max_length
                ret = {k: v[:max_length] for k, v in episode.items() if "log_" not in k}
                ret['actual_length'] = max_length * np.ones(max_length, dtype=np.int32)
                
            else:  
                if mode == 'zero':
                    ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                                mode='constant') for k, v in episode.items() if "log_" not in k}
                    ret['actual_length'] = actual_length * np.ones(max_length, dtype=np.int32)
                elif mode == 'last':
                    ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                                mode='edge') for k, v in episode.items() if "log_" not in k}
                    ret['action'][actual_length:, :6] = 0 ## zero out the first 6(pos + axis angle) action for the padded part
                    ret['actual_length'] = max_length * np.ones(max_length, dtype=np.int32)
                
                else: 
                    raise NotImplementedError
            ## for each key in ret, only keep the part of the episode from start_index
            for key in ret.keys():
                ret[key] = ret[key][start_index:]

            
            
            yield ret
class FullEpisodeSampler:
    def __init__(self):
        self.episode_index = 0
        
    def sample_episodes_with_varied_length(self, episodes, length, seed =0, max_length=1500):   
        
        while True:
            # Randomly select an episode from the dataset
            # episode_key = np_random.choice(list(episodes.keys()))
            # episode = episodes[episode_key]
            try:
                episode_key = list(episodes.keys())[self.episode_index]
            except:
                raise StopIteration
            episode = episodes[episode_key]
            # episode = next(iter(episodes.values()))
            print('episode key', episode_key) 
            self.episode_index += 1
            ## save actual length
            actual_length = len(episode["is_first"].copy())
        
            ## The original size is (B, T, D), now we need to shape into (B*T/16, 16, D)
            ## get the folded size of the episode
            # Copy the full episode data
            # ret = {k: v.copy() for k, v in episode.items() if "log_" not in k}
            ## copy the full episode data and pad the item to max_length, item can be obs, action, etc.
            ## item can be 1dim array, 2 dim array, 3 dim array, etc.
            ret = {k: v for k, v in episode.items() if "log_" not in k}
            
            
            # Ensure that the `is_first` key is set correctly for the entire episode
            if "is_first" in ret and length < max_length:
                ## set every 16th element to True in ret["is_first"]
                ret["is_first"][::length] = True
            ret['actual_length'] = actual_length * np.ones(actual_length, dtype=np.int32)
            
            yield ret
    def sample_full_episodes(self, episodes, length, seed=0, max_length=1500, mode = 'last'):
        """
        This function samples entire episodes from a given dataset of episodes. It ensures the `is_first`
        flag is correctly set for each episode.

        Args:
            episodes (dict): Dictionary containing multiple episodes. Each episode is a dictionary
                            with keys for observations, actions, etc.
            seed (int): Random seed for sampling episodes.

        Yields:
            dict: A dictionary containing the sampled full episode data, with `is_first` handled properly.
        """
        # np_random = np.random.RandomState(seed)
        
        while True:
            # Randomly select an episode from the dataset
            # episode_key = np_random.choice(list(episodes.keys()))
            # episode = episodes[episode_key]
            try:
                episode_key = list(episodes.keys())[self.episode_index]
            except:
                raise StopIteration
            episode = episodes[episode_key]
            # episode = next(iter(episodes.values()))
            print('episode key', episode_key) 
            self.episode_index += 1
            ## save actual length
            actual_length = len(episode["is_first"].copy())
            print('actual length', actual_length)
        
            ## The original size is (B, T, D), now we need to shape into (B*T/16, 16, D)
            ## get the folded size of the episode
            # Copy the full episode data
            # ret = {k: v.copy() for k, v in episode.items() if "log_" not in k}
            ## copy the full episode data and pad the item to max_length, item can be obs, action, etc.
            ## item can be 1dim array, 2 dim array, 3 dim array, etc.
            if mode == 'zero':
                ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                            mode='constant') for k, v in episode.items() if "log_" not in k}
                ret['actual_length'] = actual_length * np.ones(max_length, dtype=np.int32)
            elif mode == 'last':
                ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                            mode='edge') for k, v in episode.items() if "log_" not in k}
                ret['action'][actual_length:, :6] = 0 ## zero out the first 6(pos + axis angle) action for the padded part
                ret['actual_length'] = max_length * np.ones(max_length, dtype=np.int32)
            else: 
                raise NotImplementedError
            ## 
            
            # Ensure that the `is_first` key is set correctly for the entire episode
            if "is_first" in ret and length < max_length:
                ## set every 16th element to True in ret["is_first"]
                ret["is_first"][::length] = True
            ret['actual_length'] = actual_length * np.ones(max_length, dtype=np.int32)
            
            yield ret
    
def sample_full_episodes(episodes, length, seed=0, max_length=1500, mode='last'):
    """
    This function samples entire episodes from a given dataset of episodes. It ensures the `is_first`
    flag is correctly set for each episode.

    Args:
        episodes (dict): Dictionary containing multiple episodes. Each episode is a dictionary
                         with keys for observations, actions, etc.
        seed (int): Random seed for sampling episodes.

    Yields:
        dict: A dictionary containing the sampled full episode data, with `is_first` handled properly.
    """
    np_random = np.random.RandomState(seed)
    
    while True:
        # Randomly select an episode from the dataset
        episode_key = np_random.choice(list(episodes.keys()))
        episode = episodes[episode_key]
        ## save actual length
        actual_length = len(episode["is_first"].copy())
        print('actual length', actual_length)
        
        ## The original size is (B, T, D), now we need to shape into (B*T/16, 16, D)
        ## get the folded size of the episode
        # Copy the full episode data
        # ret = {k: v.copy() for k, v in episode.items() if "log_" not in k}
        ## copy the full episode data and pad the item to max_length, item can be obs, action, etc.
        ## item can be 1dim array, 2 dim array, 3 dim array, etc.
        if mode == 'zero':
            ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                         mode='constant') for k, v in episode.items() if "log_" not in k}
            ret['actual_length'] = actual_length * np.ones(max_length, dtype=np.int32)
        elif mode == 'last':
            ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                        mode='edge') for k, v in episode.items() if "log_" not in k}
            ret['action'][actual_length:, :6] = 0 ## zero out the first 6(pos + axis angle) action for the padded part
            ret['actual_length'] = max_length * np.ones(max_length, dtype=np.int32)
        
        else: 
            raise NotImplementedError
        
        # Ensure that the `is_first` key is set correctly for the entire episode
        if "is_first" in ret and length < max_length:
            ## set every 16th element to True in ret["is_first"]
            ret["is_first"][::length] = True
        
        
        yield ret

def sample_partial_episodes(episodes, length, seed=0, max_length=99, mode='last', start_index = 0):
    """
    This function samples part of each episodes from a given dataset of episodes. It ensures the `is_first`
    flag is correctly set for each episode.

    Args:
        episodes (dict): Dictionary containing multiple episodes. Each episode is a dictionary
                         with keys for observations, actions, etc.
        seed (int): Random seed for sampling episodes.

    Yields:
        dict: A dictionary containing the sampled full episode data, with `is_first` handled properly.
    """
    np_random = np.random.RandomState(seed)
    
    while True:
        # Randomly select an episode from the dataset
        episode_key = np_random.choice(list(episodes.keys()))
        # if isinstance(start_index, list):
        #     start_index = np_random.choice(start_index)
        #     max_length = start_index + length
        episode = episodes[episode_key]
        ## save actual length
        actual_length = len(episode["is_first"].copy())
        # print('actual length', actual_length)
        
        ## The original size is (B, T, D), now we need to shape into (B*T/16, 16, D)
        ## get the folded size of the episode
        # Copy the full episode data
        # ret = {k: v.copy() for k, v in episode.items() if "log_" not in k}
        ## copy the full episode data and pad the item to max_length, item can be obs, action, etc.
        ## item can be 1dim array, 2 dim array, 3 dim array, etc.
        if actual_length > max_length:
            actual_length = max_length
            ret = {k: v[:max_length] for k, v in episode.items() if "log_" not in k}
            ret['actual_length'] = max_length * np.ones(max_length, dtype=np.int32)
            
        else:  
            if mode == 'zero':
                ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                            mode='constant') for k, v in episode.items() if "log_" not in k}
                ret['actual_length'] = actual_length * np.ones(max_length, dtype=np.int32)
            elif mode == 'last':
                ret = {k: np.pad(v,  [(0, max_length - len(v))] + [(0, 0)] * (v[0].ndim ),
                            mode='edge') for k, v in episode.items() if "log_" not in k}
                ret['action'][actual_length:, :6] = 0 ## zero out the first 6(pos + axis angle) action for the padded part
                ret['actual_length'] = max_length * np.ones(max_length, dtype=np.int32)
            
            else: 
                raise NotImplementedError
        ## for each key in ret, only keep the part of the episode from start_index
        for key in ret.keys():
            ret[key] = ret[key][start_index:max_length]
        # Ensure that the `is_first` key is set correctly for the entire episode
        # if "is_first" in ret and length < max_length:
            ## set every 16th element to True in ret["is_first"]
            # ret["is_first"][::length] = True
        
        
        yield ret

              
def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        # num_ep = 0
        # episodes_keys = list(episodes.keys())
        # key = next(iter(episodes_keys))
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            # episode = episodes[episodes_keys[num_ep]]
            # num_ep += 1
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                # index = 0
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size

                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    if total > 0:
        print(f"Loaded {total} episodes from {directory}")
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="cuda",
    ):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255).to(device)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):
        x = self.transfwd(x)
        # x(time, batch, 1)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum(
            (self.buckets > x[..., None]).to(torch.int32), dim=-1
        )
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above

        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
            + F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1)

    def log_prob_target(self, target):
        log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)

    @classmethod
    def reduce_over_dist(cls, dist1, dist2, comparison_fn):
        """
        Create a new DiscDist that is the minimum of two distributions
        based on some criterion on the mode.
        """
        assert (
            dist1.logits.shape == dist2.logits.shape
        ), "The two distributions must have the same shape"

        mode1 = dist1.mode()
        mode2 = dist2.mode()
        mask = comparison_fn(mode1, mode2)
        new_logits = torch.where(mask, dist1.logits, dist2.logits)

        # Create a new DiscDist instance with the new logits
        return cls(
            new_logits,
            low=min(dist1.buckets[0], dist2.buckets[0]),
            high=max(dist1.buckets[-1], dist2.buckets[-1]),
            device=dist1.logits.device,
        )


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out = (
                out
                * (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            )
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out = (
                out
                * (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
            )
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)

    @classmethod
    def clone_and_detach(cls, contdist):
        if isinstance(contdist, cls):
            # Clone and detach the mean and absmax attributes
            new_mean = (
                contdist.mean.clone().detach() if contdist.mean is not None else None
            )
            new_absmax = contdist.absmax

            # Clone and detach the parameters of the underlying distribution
            new_params = {
                name: param.clone().detach()
                for name, param in contdist._dist.arg_constraints.items()
            }

            # Create a new distribution of the same type with the cloned and detached parameters
            new_dist = contdist._dist.__class__(**new_params)

            # Create a new ContDist object with the cloned and detached attributes and distribution
            new_contdist = cls(new_dist, new_absmax)
            new_contdist.mean = new_mean

            return new_contdist
        else:
            raise ValueError("Input must be a ContDist instance.")


class Bernoulli:
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


class ResidualActionWrapper:
    """
    The purpose of this class is to return the sum of the
        base_policy + residual_actions when mode() or sample()
        is called, but return the probabilities of the final
        distribution when log_prob() and entropy() is called.
    """

    def __init__(self, action_sum, residual_dist, final_discount):
        self._action_sum = action_sum
        self._residual_dist = residual_dist
        self._discount = final_discount

    def __getattr__(self, name):
        return getattr(self._residual_dist, name)

    def mode(self):
        return self._action_sum + self._discount * self._residual_dist.mode()

    def sample(self):
        return self._action_sum + self._discount * self._residual_dist.sample()

    def log_prob(self, x):
        return self._residual_dist.log_prob(x - self._action_sum)

    def entropy(self):
        return self._residual_dist.entropy()


def static_scan_for_lambda_return(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = torch.cat([outputs, last], dim=-1)
    outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = torch.flip(outputs, [1])
    outputs = torch.unbind(outputs, dim=0)
    return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    return returns


class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
        lr_decay=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if lr_decay:
            cprint(f"Scheduling {name} optimizer", attrs=["bold"], color="magenta")
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._opt,
                gamma=0.95,
            )
        else:
            self._scheduler = None

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def backward_multiple_losses(self, losses, params, retain_graph=True):
        assert all(len(loss.shape) == 0 for loss in losses), "All losses must be scalar"
        metrics = {}
        self._opt.zero_grad()
        total_loss = 0
        for i, loss in enumerate(losses):
            metrics[f"{self._name}_loss_{i}"] = loss.detach().cpu().numpy()
            scaled_loss = self._scaler.scale(loss)
            scaled_loss.backward(retain_graph=retain_graph)
            total_loss += scaled_loss
        self._scaler.unscale_(self._opt)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        self._opt.zero_grad()
        metrics[f"{self._name}_total_loss"] = total_loss.detach().cpu().numpy()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def add_new_params(self, new_params):
        self._opt.add_param_group({"params": new_params})

    def step(self):
        if self._scheduler:
            self._scheduler.step()

    def get_lr(self):
        return self._opt.param_groups[0]["lr"]

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: value.clone().unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: value.clone().unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = torch.cat(
                        [outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = torch.cat(
                                [outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def zero_weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.LayerNorm)):
        nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(torch.mean(tensor)),
        "std": to_np(torch.std(tensor)),
        "min": to_np(torch.min(tensor)),
        "max": to_np(torch.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=None
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    if visited is None:
        visited = set()
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)

def save_classifier_checkpoint(
    ckpt_name: Union[str, Callable[[int], str]],
    step: int,
    agent: Any,
    logdir: str,
)-> None:
    
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": recursively_collect_optim_state_dict(agent),
        "global_step": step
    }
    if hasattr(agent, "ema"):
        items_to_save["ema"] = agent.ema.state_dict()
    ckpt_name_str = ckpt_name(step) if callable(ckpt_name) else ckpt_name
    torch.save(items_to_save, logdir / f"{ckpt_name_str}_{step}.pt")
    print('Saving classifier model to ', logdir / f"{ckpt_name_str}_{step}.pt")
    return 

def save_checkpoint(
    ckpt_name: Union[str, Callable[[int], str]],
    step: int,
    score: float,
    best_score: float,
    agent: Any,
    logdir: str,
) -> float:
    """
    Save the current model as the last model and if the current score is better than the best score, save it as the best model.

    Args:
        ckpt_name (Callable[[int], str]): Function to generate the checkpoint name based on the step.
        step (int): Current step.
        score (float): Current score.
        best_score (float): Best score so far.
        agent (Any): The agent whose state is to be saved.
        logdir (str): Directory where to save the model.

    Returns:
        float: Updated best score.
    """
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": recursively_collect_optim_state_dict(agent),
        "global_step": step
    }
    if hasattr(agent, "ema"):
        items_to_save["ema"] = agent.ema.state_dict()
    ckpt_name_str = ckpt_name(step) if callable(ckpt_name) else ckpt_name

    # Always save the last model
    
    if (step +1) % 10000 == 0:
        torch.save(items_to_save, logdir / f"{ckpt_name_str}_{step+1}.pt")
        print("Saved last model to ", logdir / f"{ckpt_name_str}_{step+1}.pt")
    # else:
    if step == 145000:
        torch.save(items_to_save, logdir / f"{ckpt_name_str}.pt")
        print("Saved last model to ", logdir / f"{ckpt_name_str}.pt")

    # If current score is better than the best score, save the model as the best model
    # if score is not None and score <= best_score:
    #     best_score = score
    #     best_score_str = f"{best_score:.2f}".replace(".", "_")
    #     torch.save(items_to_save, logdir / f"best_{ckpt_name_str}_{best_score_str}.pt")
    #     print(
    #         "Saved best model to ", logdir / f"best_{ckpt_name_str}_{best_score_str}.pt"
    #     )

    return best_score

def save_checkpoint_classifier(
    ckpt_name: Union[str, Callable[[int], str]],
    step: int,
    score: float,
    best_score: float,
    agent: Any,
    logdir: str,
) -> float:
    """
    Save the current model as the last model and if the current score is better than the best score, save it as the best model.

    Args:
        ckpt_name (Callable[[int], str]): Function to generate the checkpoint name based on the step.
        step (int): Current step.
        score (float): Current score.
        best_score (float): Best score so far.
        agent (Any): The agent whose state is to be saved.
        logdir (str): Directory where to save the model.

    Returns:
        float: Updated best score.
    """
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": recursively_collect_optim_state_dict(agent),
    }
    if hasattr(agent, "ema"):
        items_to_save["ema"] = agent.ema.state_dict()
    ckpt_name_str = ckpt_name(step) if callable(ckpt_name) else ckpt_name

    # Always save the last model
    torch.save(items_to_save, logdir / f"{ckpt_name_str}.pt")
    print("Saved last model to ", logdir / f"classifier_{ckpt_name_str}.pt")

    # If current score is better than the best score, save the model as the best model
    if score is not None and score <= best_score:
        best_score = score
        best_score_str = f"{best_score:.2f}".replace(".", "_")
        torch.save(items_to_save, logdir / f"best_classifier_{ckpt_name_str}_{best_score_str}.pt")
        print(
            "Saved best model to ", logdir / f"best_classifier_{ckpt_name_str}_{best_score_str}.pt"
        )

    return best_score

class DreamerAgent:
    def __init__(self, agent):
        self.agent = agent
        self.reset()

    def get_action(self, obs):
        # add batch dim to all keys in obs
        obs = {k: np.array([v]) for k, v in obs.items()}
        policy_output, self.state = self.agent(
            obs, reset=None, state=self.state, training=False
        )
        policy_output = {k: v[0] for k, v in policy_output.items()}
        action = {"action": policy_output["action"].detach().cpu().numpy()}
        return action

    def reset(self):
        self.state = None


class ModelEvaluator:
    def __init__(
        self,
        config,
        agent,
        env,
        default_seed,
        visualize=False,
        parent_output_dir="",
        NUM_SEEDS=10,
        NUM_EVALS_PER_SEED=1,
    ):
        self.config = config
        self.NUM_SEEDS = NUM_SEEDS
        self.NUM_EVALS_PER_SEED = NUM_EVALS_PER_SEED
        self.default_seed = default_seed
        self.agent = agent
        self.env = env
        self.visualize = visualize
        self.parent_output_dir = parent_output_dir

    def enter_seed(self, seed):
        # Store random states for torch and numpy
        self.random_state = random.getstate()
        self.np_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()

        # Set the seeds
        set_seed_everywhere(seed)

    def exit_seed(self):
        # Reset the seeds
        set_seed_everywhere(self.default_seed)

        # Restore random states for torch and numpy
        # (To continue same sequence outside the scope of enter_seed and exit_seed)
        random.setstate(self.random_state)
        np.random.set_state(self.np_random_state)
        torch.random.set_rng_state(self.torch_random_state)

    def evaluate_agent_seed(self, seed):
        self.enter_seed(seed)

        # Make Environment and Agent
        env = self.env
        dreamerAgent = DreamerAgent(self.agent)

        # Run the Evaluation
        successes = []
        total_rewards = []
        length_episodes = []
        image_frames = []

        for _ in trange(
            self.NUM_EVALS_PER_SEED,
            desc="Evals per seed",
            position=1,
            leave=False,
            ncols=0,
        ):
            obs = env.reset()
            dreamerAgent.reset()

            total_reward_episode = 0
            done = False
            success = False
            length_episode = 0

            while not done:
                action = dreamerAgent.get_action(obs)
                obs, reward, done, _ = env.step(action)
                total_reward_episode += reward
                length_episode += 1
                if reward == 1:  # done condition for robomimic env
                    success = True
                    done = True
                if self.visualize:
                    image_frames.append(obs["agentview_image"])

            successes.append(success)
            total_rewards.append(total_reward_episode)
            length_episodes.append(length_episode)

        average_success_rate = np.mean(successes)
        average_total_reward = np.mean(total_rewards)
        average_length_episode = np.mean(length_episodes)

        # Exit the seed
        self.exit_seed()

        return (
            average_success_rate,
            average_total_reward,
            average_length_episode,
            image_frames,
        )

    def evaluate_agent(self):
        start_time = time.time()
        success_rates = []
        total_avg_rewards = []
        episode_lengths = []
        images = []
        for seed in trange(
            self.NUM_SEEDS, desc="Loop over seeds", position=0, leave=False, ncols=0
        ):
            success_rate, total_avg_reward, avg_length_episode, image_frames = (
                self.evaluate_agent_seed(seed)
            )
            success_rates.append(success_rate)
            total_avg_rewards.append(total_avg_reward)
            episode_lengths.append(avg_length_episode)
            images.extend(image_frames)

        avg_success_rate = np.mean(success_rates)
        avg_total_avg_reward = np.mean(total_avg_rewards)
        episode_length = np.mean(episode_lengths)
        print(f"Average Success Rate: {avg_success_rate}")
        print(f"Average Total Average Reward: {avg_total_avg_reward}")
        print(f"Average Episode Length: {episode_length}")
        print("Time taken:", time.time() - start_time)
        self.env.reset()

        if self.visualize:
            if self.parent_output_dir:
                video_path = f"video_avg_succ_{avg_success_rate}.mp4"
            else:
                video_path = (
                    f"{self.parent_output_dir}/video_avg_succ_{avg_success_rate}.mp4"
                )
            self.save_video(images, video_path)
        return avg_success_rate, avg_total_avg_reward, episode_length

    def save_video(self, frames, video_path):
        frames = np.array(frames)
        frame_height, frame_width, _ = frames[0].shape
        out = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height)
        )
        for frame in frames:
            out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        out.release()


def create_single_or_ensemble(ensemble_size, ensemble_subsample, name, base_kwargs):
    """Helper function to either create a single or ensemble of MLPs"""
    if ensemble_size > 1:
        cprint(
            f"Creating an ensemble of {ensemble_size} {name}s",
            color="magenta",
            attrs=["bold"],
        )
        model = networks.MLPEnsemble(
            num_models=ensemble_size,
            num_subsample=ensemble_subsample,
            **base_kwargs,
        )
    else:
        model = networks.MLP(**base_kwargs)
    return model
