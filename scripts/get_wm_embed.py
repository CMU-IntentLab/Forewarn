
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
from common.ood_utils import train_pca_kmeans, load_pca_kmeans, plot_cluster_embedding
from dreamer.dreamer import Dreamer
from dreamer.prediction import ClassifierTrainer, ClassifierLatentTrainer
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
import h5py

def main(config):
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
    step = 0
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
        logger = tools.Logger(logdir, 0, name = config.wandb_name)#config.action_repeat * step)
    logger.config(vars(config))
    logger.write() 
    
    
    ## build the training buffer
    # success_eps = collections.OrderedDict()
    if config.test_train_data:
        all_eps = collections.OrderedDict()
        empty_eps = collections.OrderedDict()
        ## get all the training episodes
    
        observation_space, action_space, _, _, _  =  tools.fill_expert_dataset_real_data_no_filtering(config, all_eps, empty_eps )
        train_dataset = make_dataset(all_eps, config)
    else: 
        success_eps = collections.OrderedDict()
        failure_eps = collections.OrderedDict()
        ## get all the training episodes
        observation_space, action_space, _, _, _  =  tools.fill_expert_dataset_real_data_no_filtering(config, success_eps, failure_eps, is_val_set=True )
        success_eval_dataset = make_dataset(success_eps, config)
        failure_eval_dataset = make_dataset(failure_eps, config)
    # # tools.fill_expert_dataset_robocasa(config, failure_eps, is_val_set=False, dataset_type='failure')
    # # all_eps = tools.merge_two_cache_dicts(success_eps, failure_eps)
    # train_dataset = make_classifier_dataset(all_eps, config)
    # validation replay buffer
    # success_val_eps = collections.OrderedDict()
    # failure_val_eps = collections.OrderedDict()
    # tools.fill_expert_dataset_real_data_for_classifier(config, success_val_eps, failure_val_eps,  is_val_set=True)
    # # observation_space, action_space, _, _, _  = tools.fill_expert_dataset_robocasa(config, success_val_eps, is_val_set=True, dataset_type='success')
    # # success_val_dataset = make_eval_dataset(success_val_eps, config)
    # # failure_val_eps = collections.OrderedDict() 
    # # tools.fill_expert_dataset_robocasa(config, failure_val_eps, is_val_set=True, dataset_type='failure')
    # # all_eps_eval = tools.merge_two_cache_dicts(success_val_eps, failure_val_eps)
    # all_eps_eval = tools.merge_two_dicts_no_relabel(success_val_eps, failure_val_eps)
    # eval_dataset = make_classifier_dataset(all_eps_eval, config)
    # failure_val_dataset = make_eval_failure_dataset(failure_val_eps, config)
    print(f"Action Space: {action_space}.") # Low: {acts.low}. High: {acts.high}")
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    # ==================== Create Agent ====================
    agent = Dreamer(
        # train_envs[0].observation_space,
        # train_envs[0].action_space,
        observation_space,
        action_space,
        config,
        logger,
        None,#success_val_dataset,
        expert_dataset= None #failure_val_dataset if config.hybrid_training else None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
        checkpoint = torch.load(config.from_ckpt)
        # step_so_far = checkpoint['total_step']
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
        # try:
        #     tools.recursively_load_optim_state_dict(
        #         agent, checkpoint["optims_state_dict"]
        #     )
        # except Exception as e:
        #     # likely due to mismatch in pretrain optimizers
        #     print("Failed to load optim state dict", e)

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
    print('Evaluating Embeddings')
    agent.eval()
    if config.test_train_data:
        eval_batch = next(iter(train_dataset))
        # eval_batch = tools.to_device(eval_batch, config.device)
        eval_latents = agent.get_latent(eval_batch, imagined_steps=63, total_steps=64, mode = 'all')
        print('Done Evaluating one batch of Embeddings from training data')
    else:
        eval_success_batch = next(iter(success_eval_dataset))
        eval_failure_batch = next(iter(failure_eval_dataset))
        # eval_success_batch = tools.to_device(eval_success_batch, config.device)
        # eval_failure_batch = tools.to_device(eval_failure_batch, config.device)
        eval_success_latents = agent.get_latent(eval_success_batch, imagined_steps=63, total_steps=64, mode = 'all')    
        eval_failure_latents = agent.get_latent(eval_failure_batch, imagined_steps=63, total_steps=64, mode = 'all')    
        breakpoint()
        print('Done Evaluating one batch of success and failure Embeddings from evaluation data')
    

    # for epoch in range(config.classifier_epoch):
    #     train_steps = classifier_trainer.train_classifier(epoch = epoch, 
    #                                         num_steps = 2* config.num_exp_trajs//config.classifier_batch_size, 
    #                                         data_loader = train_dataset,
    #                                         classifier_mode = config.classifier_mode)
    #     classifier_trainer.train_classifier(epoch = epoch,
    #                                        num_steps =train_steps,
    #                                        data_loader = eval_dataset, 
    #                                        validate=True,
    #                                        classifier_mode = config.classifier_mode)
    #     print('train_steps', train_steps)
    #     classifier_ckpt_name = "classifier"
    #     tools.save_classifier_checkpoint( classifier_ckpt_name ,agent = classifier_trainer.nets, step = train_steps, logdir=logdir)

# def make_dataset(episodes, config):
#     generator = tools.sample_episodes(episodes, config.batch_length)
#     dataset = tools.from_generator(generator, config.batch_size)
#     return dataset

def make_dataset(episodes, config):
    # generator = tools.sample_full_episodes(episodes, config.classifier_batch_length,  mode = config.classifier_padding_mode)
    sampler = tools.EvalEpisodeSampler()
    generator = sampler.sample_partial_episodes(episodes , config.classifier_batch_length,  mode = config.classifier_padding_mode, start_index = config.classifier_start_index )
    dataset = tools.from_generator(generator, 1)
    return dataset

# def make_eval_dataset(episodes, config):
#     sampler = tools.EpisodeSampler(episodes, length=config.batch_length)
#     generator = sampler.sample_episodes()
#     dataset = tools.from_eval_generator(generator, config.batch_size)
#     return dataset
# def make_eval_failure_dataset(episodes, config):
#     sampler = tools.EpisodeSpecialSampler(episodes, length=config.batch_length)
#     generator = sampler.sample_episodes()
#     dataset = tools.from_eval_generator(generator, config.batch_size)
#     return dataset

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))
def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument('--config_path', type = str, default='config.yaml')
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
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
        (pathlib.Path(sys.argv[0]).parent / "../configs" / config.config_path).read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_train_data", action="store_true")

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    final_config.time_limit = HORIZONS[final_config.task.split("_")[0]]
    print('final config test train data', final_config.test_train_data)
    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Hybrid Training: {final_config.hybrid_training}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.logdir}", "cyan", attrs=["bold"])
    main(final_config)