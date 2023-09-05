import numpy as np
from torch.distributions.categorical import Categorical
import net


class Agent():
    def __init__(self, args, envs, device, run_name):
        self.args = args
        self.critic = net.Critic(input=np.array(envs.single_observation_space.shape).prod(),
                                 output=envs.single_action_space.n,
                                 args=args,
                                 run_name=run_name,
                                 device=device
                                 ).to(device)

        self.actor = net.Actor(input=np.array(envs.single_observation_space.shape).prod(),
                               output=envs.single_action_space.n,
                               args=args,
                               run_name=run_name,
                               device=device
                               ).to(device)

    def get_value(self, x):
        values = self.critic(x)
        return values

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        values = self.critic(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), values

    def backprop(self, loss):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def save_model(self):
        self.actor.save()
        self.critic.save()

    def load_model(self, checkpoint_file):
        self.actor.load(checkpoint_file)
        self.critic.load(checkpoint_file)

    def load_current_model(self):
        self.actor.load_current()
        self.critic.load_current()


class TestAgent():
    def __init__(self, args, envs, device, run_name):
        self.args = args

        self.actor = net.Actor(input=np.array(envs.observation_space.shape).prod(),
                               output=envs.action_space.n,
                               args=args,
                               run_name=run_name,
                               device=device
                               ).to(device)

    def get_action(self, x, action=None):
        logits = self.actor(x.unsqueeze(0))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action

    def load_model(self, checkpoint_file):
        self.actor.load(checkpoint_file)

    def load_current_model(self):
        self.actor.load_current()
