import time
import arg
import ppo

if __name__ == "__main__":
    args = arg.parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.train:
        ppo.train(args,run_name=run_name)
