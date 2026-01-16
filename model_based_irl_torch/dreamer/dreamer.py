import os
import time
import numpy as np
import pathlib
import torch

from torch import nn
# import dreamer.exploration as expl
import dreamer.models as models
import dreamer.tools as tools
from common.utils import to_np, combine_dictionaries
#from diffusers.training_utils import EMAModel
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
import io
from common.ood_utils import train_pca_kmeans
# os.environ["MUJOCO_GL"] = "osmesa"


class Dreamer(nn.Module):
    def __init__(
        self, obs_space, act_space, config, logger, dataset
    ):
        super(Dreamer, self).__init__()
        self.dpi = config.size[0]
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_log_video = tools.Every(config.log_every_video)
        batch_steps = config.batch_size * config.batch_length
        self._batch_train_steps = int(
            config.steps_per_batch * config.train_ratio / batch_steps
        )
        print(
            f"Updating the agent for {self._batch_train_steps} every {config.steps_per_batch} env steps"
        )
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)

        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat if logger is not None else 0
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        # self._wm = torch.compile(self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            print("Compiling world model")
            self._wm = torch.compile(self._wm)


        if (
            config.pretrain_actor_steps > 0
            or config.pretrain_joint_steps > 0
            or config.from_ckpt is not None
        ):
            self._make_pretrain_opt()
    @classmethod
    def from_pretrained(cls, path, obs_space, act_space, config, logger, dataset, expert_dataset=None):
        ckpt = torch.load(path)
        model = cls(obs_space, act_space, config, logger, dataset, expert_dataset)
        model.load_state_dict(ckpt['agent_state_dict'])
        # model.eval()
        return model
 





   

    def pretrain_actor_model(self, data, step=None):
        """
        Data: "agentview_image", "state", "robot0_eye_in_hand_image",
              "reward", "is_first", "is_last", "is_terminal", "discount", "action", "cont"
        Shape: (batch_size x batch_length x ...)
        """
        metrics = {}
        wm = self._wm
        actor = self._task_behavior.actor

        data = wm.preprocess(data)
        if self._config.pretrain_annealing is None:
            recon_weight = 1.0
        elif self._config.pretrain_annealing == "linear":
            recon_weight = (
                self._config.pretrain_joint_steps - (step - 1)
            ) / self._config.pretrain_joint_steps
            recon_weight = max(0.0, recon_weight)
        else:
            print(self._config.pretrain_annealing)
            raise Exception("Annealing strategy must be None or Linear")

        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (
                    self._config.recon_pretrain
                    and step <= self._config.pretrain_joint_steps
                ):
                    # preds is dictionary of all all MLP+CNN keys
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = torch.zeros_like(kl_loss)

                # Zero out kl_loss and recon_loss if pretrain_bc_loss_only is True
                if self._config.pretrain_bc_loss_only == True:
                    kl_loss = torch.zeros_like(kl_loss)
                    recon_loss = torch.zeros_like(recon_loss)

                model_loss = kl_loss + recon_loss

                target = torch.Tensor(data["action"]).to(self._config.device)
                if self._config.dropout_recurrent_prob > 0:
                    feat = self._task_behavior.apply_recurrent_dropout(feat)
                action = actor(feat)
                if self._config.pretrain_loss == "mse":
                    action = action.mode()
                    actor_loss = torch.mean((action - target) ** 2)
                elif self._config.pretrain_loss == "ce":
                    actor_loss = torch.mean(-action.log_prob(target))
                else:
                    raise NotImplementedError(self._config.pretrain_loss)
                model_loss += actor_loss

            metrics = self.pretrain_opt(torch.mean(model_loss), self.pretrain_params)
            if self._config.pretrain_ema:
                self.ema.step(self._task_behavior.actor.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["actor_loss"] = to_np(actor_loss)
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["recon_weight"] = recon_weight

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics

        self._update_running_metrics(metrics)
        self._step += 1

        if self._logger is not None:
            self._maybe_log_metrics(fps_namespace="model_pretrain/")
            self._logger.step = self._step

        return metrics

    def pretrain_model_only(self, data, step=None):
        metrics = {}
        wm = self._wm
        data = wm.preprocess(data)
        if self._config.pretrain_annealing is None:
            recon_weight = 1.0
        elif self._config.pretrain_annealing == "linear":
            recon_weight = (
                self._config.pretrain_joint_steps - (step - 1)
            ) / self._config.pretrain_joint_steps
            recon_weight = max(0.0, recon_weight)
        else:
            print(self._config.pretrain_annealing)
            raise Exception("Annealing strategy must be None or Linear")

        with tools.RequiresGrad(wm):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = wm.encoder(data)
                # post: z_t, prior: \hat{z}_t
                post, prior = wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # note: kl_loss is already sum of dyn_loss and rep_loss
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)

                if (
                    self._config.recon_pretrain
                    and step <= self._config.pretrain_joint_steps
                ):
                    # preds is dictionary of all all MLP+CNN keys
                    preds = wm.heads["decoder"](feat)
                    for name, pred in preds.items():
                        loss = -pred.log_prob(data[name])
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = 0

                model_loss = kl_loss + recon_weight * recon_loss
                metrics = self.pretrain_opt(
                    torch.mean(model_loss), self.pretrain_params
                )
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["recon_weight"] = recon_weight

        with torch.cuda.amp.autocast(wm._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(wm.dynamics.get_dist(post).entropy())
            )
        metrics = {
            f"model_only_pretrain/{k}": v for k, v in metrics.items()
        }  # Add prefix model_pretrain to all metrics
        self._update_running_metrics(metrics)
        self._maybe_log_metrics(fps_namespace="model_only_pretrain/")
        self._step += 1
        self._logger.step = self._step

    def pretrain_actor_only(self, data, step=None):
        wm = self._wm
        actor = self._task_behavior.actor
        data = wm.preprocess(data)
        with tools.RequiresGrad(wm), tools.RequiresGrad(actor):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = self._wm.encoder(data)
                post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
                feat = self._wm.dynamics.get_feat(post)
                target = torch.Tensor(data["action"]).to(self._config.device)
                action = actor(feat).mean
                actor_loss = torch.mean((action - target) ** 2)
            metrics = self.pretrain_opt(torch.mean(actor_loss), self.actor_params)
            if self._config.pretrain_ema:
                self.ema.step(self.actor_params)
        metrics["actor_loss"] = to_np(actor_loss)
        metrics = {f"model_pretrain/{k}": v for k, v in metrics.items()}
        self._update_running_metrics(metrics)
        self._maybe_log_metrics(fps_namespace="model_pretrain/")
        self._step += 1
        self._logger.step = self._step

    def pretrain_regress_obs(self, data, obs_mlp, obs_opt, eval=False):
        wm = self._wm
        data = wm.preprocess(data)
        if eval:
            obs_mlp.eval()
        with tools.RequiresGrad(obs_mlp):
            with torch.cuda.amp.autocast(wm._use_amp):
                embed = self._wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])

                feat = self._wm.dynamics.get_feat(prior).detach() # want the imagined prior to be strong
                target = torch.Tensor(data["privileged_state"]).to(self._config.device)
                pred_state = obs_mlp(feat)
                obs_loss = torch.mean((pred_state - target) ** 2)
            if not eval:
                obs_opt(torch.mean(obs_loss), obs_mlp.parameters())
            else:
                obs_mlp.train()
        return obs_loss.item()

        
    ###
    def evaluate_embed(self, data):
    
        self._wm.dynamics.sample = False
        embeddings = []
        data_dict = data
        for key in data_dict.keys():
            if data_dict[key] is None:
                embeddings.append(None)
                continue
            data = self._wm.preprocess(data_dict[key])
        
            with torch.cuda.amp.autocast(self._wm._use_amp):
                embed = self._wm.encoder(data)
                ## option 1 encode the embedding with the dynamics model , posterior q(z_t| h_t, x_t), it has some history information
                ## option 2 get the imagined embedding of the future without ground truth observation prior p(z_t| h_t), 
                ## first plot the embedding with k_means clustering
                ## then plot the imagined future embedding wiht k_means clustering
                post, prior = self._wm.dynamics.observe(embed, data["action"], data["is_first"])
                ## return [z_t, h_t] (stoch, deter)
                feat = self._wm.dynamics.get_feat(post).detach()
                B, T, D = feat.size()
                ## feed in every `[k, k+T]` time step obs to get the embedding at time step `k+T`
               
                feat = feat.view(B*T, D)
                
                embeddings.append(feat)
                
        return embeddings[0], embeddings[1]
    
    def get_latent(self, data, mode='all', imagined_steps = 0, total_steps = 1):
        self._wm.dynamics.sample = False
        data = self._wm.preprocess(data)
        with torch.cuda.amp.autocast(self._wm._use_amp):
            embed = self._wm.encoder(data)
            post, prior = self._wm.dynamics.observe(embed, data["action"], data["is_first"])
            if mode == 'all':
                history_feat = self._wm.dynamics.get_feat(post)
            elif mode == 'z':
                history_feat = self._wm.dynamics.get_z(post)
            else: 
                raise NotImplementedError
            init = {k: v[:, -1] for k, v in post.items()}
            prior = self._wm.dynamics.imagine_with_action(data["action"], init)
            if mode == 'all':
                imagined_feat = self._wm.dynamics.get_feat(prior)[:, :imagined_steps]
            elif mode == 'z':
                imagined_feat = self._wm.dynamics.get_z(prior)[:, :imagined_steps]
        feat = torch.cat([history_feat, imagined_feat], dim=1)[:,:total_steps]
        return feat
                    
                    
           
                
        

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self, video_pred_log=False, fps_namespace=""):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True, fps_namespace=fps_namespace)

    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.pretrain_actor_steps + config.pretrain_joint_steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
            }
            if config.recon_pretrain:
                model_params["params"] += list(self._wm.heads["decoder"].parameters())
            # actor_params = {
            #     "params": list(self._task_behavior.actor.parameters()),
            #     "lr": config.actor["lr"],
            #     "eps": config.actor["eps"],
            #     "clip": config.actor["grad_clip"],
            # }
            self.pretrain_params = list(model_params["params"]) 
            #+ list(
            #    actor_params["params"]
            #)
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params, ], **standard_kwargs
            ) #actor_params
            # self.actor_params = list(self._task_behavior.actor.parameters())
            if config.pretrain_ema:
                self.ema = EMAModel(
                    parameters=self.actor_params,
                    decay=config.ema_decay,
                    power=config.ema_power,
                )
                print("EMA")
            print(
                f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables."
            )

    def _reward_fn(self, feat, state, action):
        return self._wm.heads["reward"](self._wm.dynamics.get_feat(state)).mode()
