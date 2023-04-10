import argparse
from distutils.util import strtobool
import os

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo",#os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=1.e-0,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--stdp", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled,  between stdp and backprop")
    
    # Network specific arguments
    parser.add_argument('--actor-shape',type=int,default=[64,64],nargs='+',
        help='the shape of actor`s hidden layers')
    parser.add_argument('--critic-shape',type=int,default=[64,64],nargs='+',
        help="the shape of critic`s hidden layers")
    parser.add_argument("--save", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggle to not save new model")
    # STDP
    parser.add_argument("--stdp-A-plus", type=float, default=0.3125,
        help="stdp alpha plus coefficient")
    parser.add_argument("--stdp-A-minus", type=float, default=0.3125,
        help="stdp alpha coefficient")
    parser.add_argument("--stdp-tau-plus", type=float, default=3.2,
        help="snn tau plus coefficient")
    parser.add_argument("--stdp-tau-minus", type=float, default=3.2,
        help="snn tau minus coefficient")
    parser.add_argument("--stdp-weight-threshold", type=float, default=3.0,
        help="snn tau coefficient")
    # SNN
    parser.add_argument("--snn-a", type=float, default=0.02,
        help="snn alpha coefficient")
    parser.add_argument("--snn-b", type=float, default=0.2,
        help="snn beta coefficient")
    parser.add_argument("--snn-c", type=float, default=0.65,
        help="snn alpha coefficient")
    parser.add_argument("--snn-d", type=float, default=0.02,
        help="snn beta coefficient")
    parser.add_argument("--snn-threshold", type=float, default=1.0,
        help="snn threshold coefficient")
    parser.add_argument("--snn-num-steps", type=int, default=10,
        help="timesteps of the snn simulation")
    
    # Mode specific arguments
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="toggle to not train new model")
    parser.add_argument("--load-model", type=str, default=None,
        help="load old model")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args