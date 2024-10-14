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

# def train_lx(ckpt_name, log_dir):
#     recon_steps = 2501
#     best_pretrain_success_classifier = float("inf")
#     lx_mlp, lx_opt = agent._wm._init_lx_mlp(config, 1)
#     train_loss = []
#     eval_loss = []
#     for i in range(recon_steps):
#         if i % 250 == 0:
#             print('eval')
#             new_loss, eval_plot = agent.train_lx(
#                 next(obs_eval_dataset), lx_mlp, lx_opt, eval=True
#             )
#             eval_loss.append(new_loss)
#             logger.image("classifier", np.transpose(eval_plot, (2, 0, 1)))
#             logger.write(step=i+40000)
#             best_pretrain_success_classifier = tools.save_checkpoint(
#                 ckpt_name, i, new_loss, best_pretrain_success_classifier, lx_mlp, logdir
#             )

#         else:
#             new_loss, _ = agent.train_lx(
#                 next(obs_train_dataset), lx_mlp, lx_opt
#             )
#             train_loss.append(new_loss)
#     log_plot("train_lx_loss", train_loss)
#     log_plot("eval_lx_loss", eval_loss)
#     logger.scalar("pretrain/train_lx_loss_min", np.min(train_loss))
#     logger.scalar("pretrain/eval_lx_loss_min", np.min(eval_loss))
#     logger.write(step=i)
#     print(eval_loss)
#     print('logged')
#     return lx_mlp, lx_opt
def main():
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
        logger = tools.Logger(logdir, config.action_repeat * step)
    logger.config(vars(config))
    logger.write() 
    # validation replay buffer
    success_val_eps = collections.OrderedDict()
    observation_space, action_space, _, _, _  = tools.fill_expert_dataset_robocasa(config, success_val_eps, is_val_set=True, dataset_type='success')
    success_val_dataset = make_dataset(success_val_eps, config)
    failure_val_eps = collections.OrderedDict() 
    tools.fill_expert_dataset_robocasa(config, failure_val_eps, is_val_set=True, dataset_type='failure')
    failure_val_dataset = make_dataset(failure_val_eps, config)
    # ==================== Create Agent ====================
    agent = Dreamer(
        # train_envs[0].observation_space,
        # train_envs[0].action_space,
        observation_space,
        action_space,
        config,
        logger,
        success_dataset,
        expert_dataset=failure_dataset if config.hybrid_training else None,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if config.from_ckpt and Path(config.from_ckpt).exists():
        print(f"Loading ckpt from {config.from_ckpt}")
        # checkpoint = torch.load(config.from_ckpt)
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
        
        success_data = next(success_val_dataset)
        failure_data = next(failure_val_dataset)
        agent.evaluate_embed({"success": success_data, "failure": failure_data})