import numpy as np
from torch.distributions.categorical import Categorical
import net
import torch.nn as nn
class Agent():
    def __init__(self,args,envs,device,run_name):
        self.args = args
        self.critic = net.Critic(input=np.array(envs.single_observation_space.shape).prod(),
                                output=envs.single_action_space.n,
                                args=args,
                                run_name=run_name
                                ).to(device)
        
        self.actor = net.Actor(input=np.array(envs.single_observation_space.shape).prod(),
                                output=envs.single_action_space.n,
                                args=args,
                                run_name=run_name
                                ).to(device)


    def get_value(self, x):
        values, _ = self.critic(x)
        return values 
    
    def get_action_and_value(self, x, action=None):
        logits, actor_spks = self.actor(x)
        values, critic_spks = self.critic(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), values, actor_spks, critic_spks
    
    def backprop(self,loss):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def stdp(self,actor_output,critic_output,reward,batch_size):
            self.actor.r_stdp(actor_output,reward,batch_size)
            self.critic.r_stdp(critic_output,reward,batch_size)

    def reset_actor_critic_traces(self):
        self.actor.reset_traces()
        self.critic.reset_traces()         

     
    def save_model(self):
        self.actor.save()
        self.critic.save()



    def load_model(self,checkpoint_file):
        self.actor.load(checkpoint_file)    