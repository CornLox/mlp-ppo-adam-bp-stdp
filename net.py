import numpy as np
import torch
import torch.nn as nn
import os
from torch.nn.utils.prune import l1_unstructured, remove,is_pruned
from torch import optim
import snn

def tospikes(x):
    x =  (np.array(x)+1.0)/2.0  #map from [-1,1] to [0,1]
    x = np.greater_equal(x, np.random.rand(x.shape[0],x.shape[1]))
    x = torch.Tensor(x).float()
    return x
       

def layer_init(layer, std=np.sqrt(2), bias_const=0.0): #initialization
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Network(nn.Module):
    def __init__(self,args,run_name):
        super(Network,self).__init__()
        self.checkpoint_file = f"{run_name}"
        self.checkpoint_dir = ""
        self.args = args
        self.modules_list = nn.ModuleList()
        self.initial_dw_plus = []
        self.initial_dw_minus = []
        self.initial_db_plus =[]
        self.initial_db_minus = []
        self.dw_plus = []
        self.dw_minus =[]
        self.db_plus = []
        self.db_minus =[]
        
     
    def delta_W(self,t_post,t_pre,index):
        A_plus, A_minus, tau_plus, tau_minus = self.args.stdp_A_plus, self.args.stdp_A_minus, self.args.stdp_tau_plus, self.args.stdp_tau_minus
        dt = 1/ self.args.snn_num_steps
        self.dw_plus[index] = -self.dw_plus[index]/tau_plus
        self.dw_minus[index] = -self.dw_minus[index]/tau_minus
        # STDP change
        t_post = torch.squeeze(t_post).tolist()
        t_pre = torch.squeeze(t_pre).tolist()
        for j, _ in enumerate(t_pre):      
            self.dw_plus[index][:,j] += t_pre[j]*A_plus
        if type(t_post) != float:  
            for i, _ in enumerate(t_post):
                self.dw_minus[index][i,:] += t_post[i]*A_minus
        else:    
             self.dw_minus[index] += t_post*A_minus
        self.dw_plus[index] *= dt
        self.dw_minus[index] *= dt     

    def delta_B(self,t_post,index):
        A_plus, A_minus, tau_plus, tau_minus = self.args.stdp_A_plus, self.args.stdp_A_minus, self.args.stdp_tau_plus, self.args.stdp_tau_minus
        dt = 1/ self.args.snn_num_steps
        # STDP change
        t_post = torch.squeeze(t_post)
        t_pre = torch.ones_like(t_post)
        self.db_plus[index] = -self.db_plus[index]/tau_plus+t_pre*A_plus
        self.db_minus[index] = -self.db_minus[index]/tau_minus+t_post*A_minus
        self.db_plus[index] *= dt
        self.db_minus[index] *= dt
                  
    def set_weights(self,index,reward):
        thr = self.args.stdp_weight_threshold
        dw = reward*(self.dw_plus[index] - self.dw_minus[index])
        with torch.no_grad():
            self.modules_list[2*index].weight = nn.Parameter(torch.clamp(self.modules_list[2*index].weight+dw,min=-thr,max=thr)) 

    def set_bias(self,index,reward):
        thr = self.args.stdp_weight_threshold
        db = reward*(self.db_plus[index] - self.db_minus[index])
        with torch.no_grad():  
            self.modules_list[2*index].bias = nn.Parameter(torch.clamp(self.modules_list[2*index].bias+db,min=-thr,max=thr))                

    
    def r_stdp(self,state, reward,batch_size):
        
        for index in range(len(self.modules_list)//2):
                for i in range(self.args.snn_num_steps):    
                    for j in range(batch_size):
                        if reward != 0.0:
                            self.delta_W(state[index + 1][i][j],state[index][i][j],index)
                            self.set_weights(index,reward)
                            self.delta_B(state[index + 1][i][j],index)
                            self.set_bias(index,reward)                          

    def add_prunning(self):
        if self.args.amount_prunned > 0:    
            for _, module in enumerate(self.modules_list):
                if is_pruned(module):
                    weights = module.weight
                    remove(module=module,name='weight')
                    with torch.no_grad():
                        module.weight= weights        
                l1_unstructured(module=module,name='weight',amount=self.args.amount_prunned)
        else:
            return        

    def remove_prunning(self):
        if self.args.amount_prunned > 0:
            for _, module in enumerate(self.modules_list):
                if is_pruned(module):
                    weights = module.weight
                    remove(module=module,name='weight')
                    with torch.no_grad():
                        module.weight= weights
        else: 
            return                      

    def forward(self, input):
        spk_rec = [[]   for _ in range(len(self.modules_list) // 2 +1) ]
        for _ in range(self.args.snn_num_steps):
            for index, module in enumerate(self.modules_list):
                if index == 0:
                    spk = tospikes(input)
                    spk_rec[0].append(spk)  
                    spk = module(spk)
                elif index % 2 == 0:  
                    spk = module(spk)
                else: #index % 2 == 1:
                    spk = module(spk)
                    spk_rec[index //2 + 1].append(spk)
        output = torch.stack(spk_rec[-1], dim=0)
        output = output.sum(dim=0)/self.args.snn_num_steps
        return output, spk_rec

    def save(self):
        self.remove_prunning()
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)    
        torch.save(self.state_dict(),f"{self.checkpoint_dir}/{self.checkpoint_file}")
       
        
    def load(self,checkpoint_file):
        self.load_state_dict(torch.load(f"{self.checkpoint_dir}/{checkpoint_file}"))

    def load_current(self):
        self.load_state_dict(torch.load(f"{self.checkpoint_dir}/{self.checkpoint_file}"))    

class Actor(Network):
    def __init__(self,input,output,args,run_name):
        super(Actor,self).__init__(args,run_name)
        self.checkpoint_dir = "models/actor"
        hidden = args.actor_shape
        if len(hidden)>0: 
            self.modules_list.append(layer_init(nn.Linear(input,hidden[0])))
            self.initial_dw_plus.append(torch.zeros((hidden[0],input)))
            self.initial_dw_minus.append(torch.zeros((hidden[0],input)))
            self.initial_db_plus.append(torch.zeros(hidden[0]))
            self.initial_db_minus.append(torch.zeros(hidden[0]))
            self.modules_list.append(snn.Izi(hidden[0],args))
            for i in range(0, len(hidden)-1):
                self.modules_list.append(layer_init(nn.Linear(hidden[i],hidden[i+1])))
                self.initial_dw_plus.append(torch.zeros((hidden[i+1],hidden[i])))
                self.initial_dw_minus.append(torch.zeros((hidden[i+1],hidden[i])))
                self.initial_db_plus.append(torch.zeros(hidden[i+1]))
                self.initial_db_minus.append(torch.zeros(hidden[i+1]))
                self.modules_list.append(snn.Izi(hidden[i+1],args))
            self.modules_list.append(layer_init(nn.Linear(hidden[-1],output),std=0.01)) #low std to make the prob of taking each action similar
            self.initial_dw_plus.append(torch.zeros((output,hidden[-1])))
            self.initial_dw_minus.append(torch.zeros((output,hidden[-1])))
            self.initial_db_plus.append(torch.zeros(output))
            self.initial_db_minus.append(torch.zeros(output))
            self.modules_list.append(snn.Izi(output,args))          
        else:
            self.modules_list.append(layer_init(nn.Linear(input,output),std=0.01)) #low std to make the prob of taking each action similar
            self.initial_dw_plus.append(torch.zeros((output,input)))
            self.initial_dw_minus.append(torch.zeros((output,input)))
            self.initial_db_plus.append(torch.zeros(output))
            self.initial_db_minus.append(torch.zeros(output))
            self.modules_list.append(snn.Izi(output,args))
        self.network = nn.Sequential(*self.modules_list)
        #self.logstd = nn.Parameter(torch.zeros(1,output))
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=1e-5)
        self.reset_traces()


    def reset_traces(self):
        self.dw_plus = [s.clone() for s in self.initial_dw_plus]
        self.dw_minus = [s.clone() for s in self.initial_dw_minus]
        self.db_plus = [s.clone() for s in self.initial_db_plus]
        self.db_minus = [s.clone() for s in self.initial_db_minus]    

        


class Critic(Network):
    def __init__(self,input,output,args,run_name):
        super(Critic,self).__init__(args,run_name)
        self.checkpoint_dir = "models/critic"
          
        hidden = args.critic_shape
        if len(hidden)>0: 
            self.modules_list.append(layer_init(nn.Linear(input,hidden[0])))
            self.initial_dw_plus .append(torch.zeros((hidden[0],input)))
            self.initial_dw_minus .append(torch.zeros((hidden[0],input,)))
            self.initial_db_plus .append(torch.zeros(hidden[0]))
            self.initial_db_minus .append(torch.zeros(hidden[0]))
            self.modules_list.append(snn.Izi(hidden[0],args))
            for i in range(0, len(hidden)-1):
                self.modules_list.append(layer_init(nn.Linear(hidden[i],hidden[i+1])))
                self.initial_dw_plus.append(torch.zeros((hidden[i+1],hidden[i])))
                self.initial_dw_minus.append(torch.zeros((hidden[i+1],hidden[i])))
                self.initial_db_plus.append(torch.zeros(hidden[i+1]))
                self.initial_db_minus.append(torch.zeros(hidden[i+1]))
                self.modules_list.append(snn.Izi(hidden[i+1],args))
            self.modules_list.append(layer_init(nn.Linear(hidden[-1],1),std=1.)) #low std to make the prob of taking each action similar
            self.initial_dw_plus .append(torch.zeros((1,hidden[-1])))
            self.initial_dw_minus.append(torch.zeros((1,hidden[-1])))
            self.initial_db_plus .append(torch.zeros(1))
            self.initial_db_minus.append(torch.zeros(1))
            self.modules_list.append(snn.Izi(1,args))
        else:
            self.modules_list.append(layer_init(nn.Linear(input,1),std=1.)) #low std to make the prob of taking each action similar
            self.initial_dw_plus.append(torch.zeros((1,input)))
            self.initial_dw_minus.append(torch.zeros((1,input)))
            self.initial_db_plus .append(torch.zeros(1))
            self.initial_db_minus.append(torch.zeros(1))
            self.modules_list.append(snn.Izi(1,args))
        self.network = nn.Sequential(*self.modules_list)
        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=1e-5)
        self.reset_traces()

    def reset_traces(self):
        self.dw_plus = [s.clone() for s in self.initial_dw_plus]
        self.dw_minus = [s.clone() for s in self.initial_dw_minus]
        self.db_plus = [s.clone() for s in self.initial_db_plus]
        self.db_minus = [s.clone() for s in self.initial_db_minus]    






    
       


