from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import BranchingNet


class HGQNr2SumMixPolicy(DQNPolicy):
    """Implementation of the Branching dual Q network arXiv:1711.08946.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double network. Default to True.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: BranchingNet,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 0,
            reward_normalization: bool = False,
            is_double: bool = True,
            original_action_dim: int = 3,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model, optim, discount_factor, estimation_step, target_update_freq,
            reward_normalization, is_double
        )
        assert estimation_step == 1, "N-step bigger than one is not supported by BDQ"
        self.action_per_branch = model.action_per_branch
        # self.num_branches = model.num_branches
        self.original_action_dim = original_action_dim
        # self.mix_net = torch.nn.Linear(self.num_branches, 1)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            # act = np.expand_dims(self(batch, input="obs_next").act, -1)
            # act = to_torch(act, dtype=torch.long, device=target_q.device)
            act_index = np.expand_dims(self(batch, input="obs_next").act_index, -1)
            act_index = to_torch(act_index, dtype=torch.long, device=target_q.device)
        else:
            act_index = target_q.max(-1).indices.unsqueeze(-1)
        # return torch.gather(target_q, -1, act).squeeze()
        return torch.gather(target_q, -1, act_index).squeeze()

    def _compute_return(
            self,
            batch: Batch,
            buffer: ReplayBuffer,
            indice: np.ndarray,
            gamma: float = 0.99,
    ) -> Batch:
        rew = batch.rew
        with torch.no_grad():
            target_q_torch = self._target_q(buffer, indice)  # (bsz, ?)

        target_q = to_numpy(target_q_torch)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        end_flag = end_flag[indice]
        # mix net
        # mean_target_q = np.mean(target_q, -1) if len(target_q.shape) > 1 else target_q  # 原论文公式6
        # print("=======_compute_return========")
        # print('target_q.shape', target_q.shape) # [512, 3+3]
        # print('rew.shape', rew.shape) # [512, ]

        _target_q = rew + gamma * target_q * (1 - end_flag)
        target_q = _target_q
        # [batch_size, |A|]==[512, 125]

        # target_q = np.repeat(_target_q[..., None], self.num_branches, axis=-1)
        # target_q = np.repeat(target_q[..., None], self.action_per_branch, axis=-1)  # action_per_branch = num_of_bins

        batch.returns = to_torch_as(target_q, target_q_torch)

        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def process_fn(
            self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the 1-step return for BDQ targets."""
        return self._compute_return(batch, buffer, indices)

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[Dict, Batch, np.ndarray]] = None,
            model: str = "model",
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        # logits, value, hidden = model(obs_next, state=state, info=batch.info)
        # logits.shape: [512,3]
        q_values, order_1_values, order_2_values, value_out, hidden = model(obs_next, state=state, info=batch.info)

        # print("=======_compute_return========")
        # print('target_q.shape', target_q.shape) # [512, 3+3]
        # print('rew.shape', rew.shape) # [512, ]

        act_index = to_numpy(q_values.max(dim=-1)[1])

        from itertools import product
        # disc_to_cont: transform discrete action index to original continuous action
        self.action_per_branch = model.action_per_branch
        # self.num_branches = model.num_branches
        self.K = self.action_per_branch ** self.original_action_dim
        self.disc_to_cont = list(
            product(*[list(range(self.action_per_branch)) for _ in range(self.original_action_dim)]))

        act = to_numpy([[k for k in self.disc_to_cont[int(action)]] for action in act_index])

        return Batch(logits=q_values, act=act, act_index=act_index, state=hidden)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        # act = to_torch(batch.act, dtype=torch.long, device=batch.returns.device)
        act = batch.act  # shape 512,3
        act_index = np.array([action[0] * self.action_per_branch ** 2 + action[1] * self.action_per_branch ** 1 +
                              action[1] * self.action_per_branch ** 0 for action in act])

        act_index = to_torch(act_index, dtype=torch.long, device=batch.returns.device)

        q = self(batch).logits  # [512,5**3]

        # print("=======learn========")
        # print('act.shape', act.shape)   # [512,3]
        # print('q.shape', q.shape)  # [512,3,5]

        act_mask = torch.zeros_like(q)
        # act_mask = act_mask.scatter_(-1, act.unsqueeze(-1), 1)
        act_mask = act_mask.scatter_(-1, act_index.unsqueeze(-1), 1)

        # print('act_mask.shape', act_mask.shape)  # [512,3,5]

        # only the selected action bin in each dim is non-zero
        act_q = q * act_mask  # [512, 3, 5]
        # act_q.sum(-1) is the selected action-q value
        act_q = to_torch(act_q.sum(-1))  # [512, 1]

        returns = batch.returns

        # print('returns.shape', returns.shape)  # [512,3,5]
        # returns = returns * act_mask
        # # print('returns.shape', returns.shape)  # [512,3,5]
        # # print('act_q.shape', act_q.shape)  # [512,3,5]
        td_error = returns - act_q

        # td_error = returns - mix_act_q

        loss = (td_error.pow(2).sum(-1).mean(-1) * weight).mean()
        batch.weight = td_error.sum(-1).sum(-1)  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(
            self,
            act: Union[np.ndarray, Batch],
            batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            rand_act = np.random.randint(
                low=0, high=self.action_per_branch, size=(bsz, act.shape[-1])
            )
            if hasattr(batch.obs, "mask"):
                rand_act += batch.obs.mask
            act[rand_mask] = rand_act[rand_mask]
        return act
