import argparse
import collections
import copy
import warnings
import functools
import time
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
saferl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../safety_rl'))
sys.path.append(saferl_dir)
print(sys.path)

# from safety_rl.gym_reachability import gym_reachability  # Custom Gym env.


import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint
from torch import distributions as torchd
from tqdm import trange
import torch.multiprocessing as mp
import dreamer.tools as tools
# import envs.wrappers as wrappers
from common.constants import HORIZONS, IMAGE_OBS_KEYS
from common.utils import (
    create_shape_meta,
    get_robomimic_dataset_path_and_env_meta,
    to_np,
    combine_dictionaries,
)
from dreamer.dreamer import Dreamer
# from dreamer.parallel import Damy, Parallel
#from environments.env_make import make_env_robomimic
from termcolor import cprint
from functools import partial
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from dreamer.tools import ModelEvaluator
# os.environ["MUJOCO_GL"] = "osmesa"
from gym import spaces
from gym.spaces import Discrete 
import gym


import torch


from concurrent.futures import ThreadPoolExecutor

def mixed_success_failure_sample(
    batch_size, success_dataset, failure_dataset, device, remove_obs_stack=False
):
    """
    Sample 50% from expert dataset and 50% from failure dataset.
    If remove_obs_stack is True, keep only the latest obs in the batch.
    """
    assert batch_size % 2 == 0, "Batch Size should be even."

    # Sample from the datasets
    success_batch = next(success_dataset)
    failure_batch = next(failure_dataset)

    data_batch = {}

    def process_key(key):
        # Convert success and failure batches to PyTorch tensors and stack them directly
        stacked_data = torch.cat(
            [torch.tensor(success_batch[key], dtype=torch.float32, device=device),
             torch.tensor(failure_batch[key], dtype=torch.float32, device=device)],
            dim=0
        )
        
        return key, stacked_data

    # Use ThreadPoolExecutor to parallelize the for loop
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_key, key) for key in success_batch.keys()]
        for future in futures:
            key, stacked_data = future.result()
            data_batch[key] = stacked_data

    if remove_obs_stack:
        data_batch = remove_obs_stack(data_batch)

    return data_batch


def train_eval(config):
    mp.set_start_method('spawn', force=True) 
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    # ==================== Logging ====================
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(f"{logdir}/config.yaml", "w") as f:
        yaml.dump(vars(config), f)
    # step in logger is environmental step
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    # step = count_steps(config.traindir)
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
        checkpoint = torch.load(config.from_ckpt)
        step = checkpoint['global_step']
    elif config.from_ckpt and not Path(config.from_ckpt).exists():
        print(f"folder {config.from_ckpt} not exists")
    else: 
        step = count_steps(config.traindir)
    if config.debug:
        logger = tools.DebugLogger(logdir, config.action_repeat * step)
    else:
        logger = tools.Logger(logdir, config.action_repeat * step, name = config.wandb_name)
    logger.config(vars(config))
    logger.write()

    # ==================== Create dataset ====================
    # replay buffer
    all_eps = collections.OrderedDict()
  
    observation_space, action_space, _, _, _ = tools.fill_expert_dataset_real_data(config, all_eps)
    all_dataset = make_dataset(all_eps, config)
    val_eps = collections.OrderedDict()
    tools.fill_expert_dataset_real_data(config, val_eps, is_val_set=True)
    val_dataset = make_dataset(val_eps, config)

    

    # ==================== Create envs ====================
    


    # == CONFIGURATION ==
    
    env_name = config.task + '_img'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # == Environment ==
    print("\n== Environment Information ==")

    print('env_name', env_name)
 


    bounds = np.array([[-1.1, 1.1], [-1.1, 1.1], [0, 2 * np.pi]])



    print(f"Action Space: {action_space}.")# Low: {acts.low}. High: {acts.high}")
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    # ==================== Create Agent ====================
    agent = Dreamer(

        observation_space,
        action_space,
        config,
        logger,
        all_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
      
        if config.critic_ensemble_size > 1 or config.reward_ensemble_size > 1:
            # only false if loading with different ensemble size
            mk, uk = agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
            for k in mk:
                assert (
                    "Value" in k
                    or "_slow_value" in k
                    or "value" in k
                    or "Reward" in k
                    or "reward" in k
                )
            for k in uk:
                assert (
                    "Value" in k
                    or "_slow_value" in k
                    or "value" in k
                    or "Reward" in k
                    or "reward" in k
                )
        else:
            agent.load_state_dict(checkpoint["agent_state_dict"])
        try:
            tools.recursively_load_optim_state_dict(
                agent, checkpoint["optims_state_dict"]
            )
        except Exception as e:
            # likely due to mismatch in pretrain optimizers
            print("Failed to load optim state dict", e)

        pretrained_ckpt_config = Path(config.from_ckpt).parent / "config.yaml"
        if pretrained_ckpt_config.exists():
            pretrained_config = yaml.load(pretrained_ckpt_config.read_text())
            if pretrained_config["num_exp_trajs"] == config.num_exp_trajs:
                warnings.warn(
                    f"Mismatch in number of expert trajectories in pretrained config {pretrained_config['num_exp_trajs']} and actual {config.num_exp_trajs}"
                )

        agent._should_pretrain._once = False
        if config.from_ckpt and config.pretrain_ema:
            print("Using EMA weights from pretraining")
            agent.ema.load_state_dict(checkpoint["ema"])
            agent.ema.copy_to(agent._task_behavior.actor.parameters())
    
    # ==================== Training Fns ====================
    def log_plot(title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        logger.image("pretrain/" + title, np.transpose(plot_arr, (2, 0, 1)))

    

    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        print(
            f"Evaluating for Seeds: {config.eval_num_seeds} and Evals per seed: {config.eval_per_seed}"
        )
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if config.video_pred_log:
            '''_, _ = tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=1,
                eval_prefix=eval_prefix,
            )'''
            video_pred, loss = agent._wm.video_pred(next(val_dataset))
            
            logger.video("eval_recon/openl_agent", to_np(video_pred))
            for key in loss.keys():
                logger.scalar(f'eval_recon/{key}', float(torch.mean(loss[key]).cpu().numpy()))
            

            if other_dataset:
                
               
                video_pred, loss = agent._wm.video_pred(next(other_dataset))
               
                logger.video("train_recon/openl_agent", to_np(video_pred))
                for key in loss:
                    logger.scalar(f'train_recon/{key}', float(torch.mean(loss[key]).cpu().numpy()))
           

        # Get stats
        '''eval_success, eval_score, eval_ep_len = ModelEvaluator(
            config=config,
            agent=eval_policy,
            env=eval_envs[0]._env,
            default_seed=config.seed,
            NUM_SEEDS=config.eval_num_seeds,
            NUM_EVALS_PER_SEED=config.eval_per_seed,
        ).evaluate_agent()'''

        # Update the metrics in the logger
     
        logger.scalar(
            f"{eval_prefix}/eval_episodes", config.eval_num_seeds * config.eval_per_seed
        )
        logger.write(step=logger.step)
       
        total_loss = 0
        for key in loss.keys():
            total_loss += np.mean(float(torch.mean(loss[key]).cpu().numpy()))
   
        return  total_loss, loss
      



    # ==================== Actor Pretrain ====================
    total_pretrain_steps = config.pretrain_joint_steps + config.pretrain_actor_steps
    print(total_pretrain_steps)
    if total_pretrain_steps > 0:
        assert not (config.pretrain_on_random and config.pretrain_on_random_mixed)
        if config.pretrain_on_random or config.pretrain_on_random_mixed:
            assert (
                config.offline_traindir is not None
            ), "Need to load in random data to be trained"
        cprint(
            f"Pretraining for {config.pretrain_joint_steps=}, {config.pretrain_actor_steps=}",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = (  # noqa: E731
            lambda step: f"pretrain_joint"
            if step < config.pretrain_joint_steps
            else "pretrain_actor"
        )
        best_pretrain_success = float("inf")
        for step in trange(
            total_pretrain_steps,
            desc="World Model pretraining",
            ncols=0,
            leave=False,
        ):
            
            if (
                config.eval_num_seeds > 0
                and ((step + 1) % config.eval_every) == 0
                or step == 1
    
            ):
               
                print('eval')
                score, success = evaluate(
                    other_dataset=all_dataset, eval_prefix="pretrain"
                )
                tools.save_checkpoint(
                            ckpt_name, step, success, best_pretrain_success, agent, logdir
                        )
            
           

          
           
            sample = next(all_dataset)
       
            agent.pretrain_model_only(sample, step)
        



    exit()


    


def close_envs(envs):
    for env in envs:
        try:
            env.close()
        except Exception:
            pass


def possibly_get_expert_data(
    expert_dataset, config, step, model_only_steps, joint_steps
):
    """Helper function to decide when to get the expert dataset,
    preventing unnecessary calls to next(expert_dataset)"""
    bc_loss_cond = (
        config.train_residuals or config.bc_reg
    ) and step >= model_only_steps

    # TODO: VERIFY CORRECTNESS
    hybrid_fitting_cond = (
        config.hybrid_training and step < model_only_steps + joint_steps
    )

    if bc_loss_cond or hybrid_fitting_cond:
        return next(expert_dataset)
    return None


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


# def make_env(config):
#     suite, task = config.task.split("_", 1)
#     assert suite == "robomimic", f"Unknown suite {suite}"
#     assert task in HORIZONS.keys(), f"Unknown task {task}"
#     dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
#         env_id=str(task).upper(),
#         shaped=config.shape_rewards,
#         image_size=config.image_size,
#         done_mode=config.done_mode,
#     )
#     shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)

#     shape_rewards = config.shape_rewards
#     shift_rewards = config.shift_rewards
#     env = make_env_robomimic(
#         env_meta,
#         IMAGE_OBS_KEYS,
#         shape_meta,
#         add_state=True,
#         reward_shift_wrapper=shift_rewards,
#         reward_shaping=shape_rewards,
#         offscreen_render=False,
#     )
#     env = wrappers.TimeLimit(env, config.time_limit)
#     env = wrappers.SelectAction(env, key="action")
#     env = wrappers.UUID(env)
#     return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    parser.add_argument('--config_path', type = str, default='config.yaml')

    config, remaining = parser.parse_known_args()

    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = (
            f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        )
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        # (pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
        (pathlib.Path(sys.argv[0]).parent / "../configs" / config.config_path).read_text()

    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    final_config.time_limit = HORIZONS[final_config.task.split("_")[0]]

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Hybrid Training: {final_config.hybrid_training}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    print("---------------------")

    train_eval(final_config)
