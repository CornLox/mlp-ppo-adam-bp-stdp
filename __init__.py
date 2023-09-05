import time
import arg
import ppo

if __name__ == "__main__":
    args = arg.parse_args()
    if args.load_model is not None:
        run_name = args.load_model
    else:
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.train:
        ppo.train(args=args, run_name=run_name)
    ppo.test(args=args, run_name=run_name)
