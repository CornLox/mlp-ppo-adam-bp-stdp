from torch.utils.tensorboard import SummaryWriter
import time

class Writer(SummaryWriter):
    def __init__(self,args,run_name):
        super(Writer,self).__init__(log_dir=f"runs/{run_name}")
        self.start_time = time.time()
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    def episode_summary(self,global_step,
                              episodic_return,
                              episodic_length):
        print(f"global_step={global_step}, episodic_return={episodic_return}")
        self.add_scalar("charts/episodic_return", episodic_return, global_step)
        self.add_scalar("charts/episodic_length", episodic_length, global_step)    

    def final_summary(self,global_step,
                      learning_rate,
                      value_loss,
                      policy_loss,
                      entropy,
                      old_approx_kl,
                      approx_kl,
                      clipfrac,
                      explained_var
                      ):
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.add_scalar("charts/learning_rate", learning_rate, global_step)
        self.add_scalar("losses/value_loss", value_loss, global_step)
        self.add_scalar("losses/policy_loss", policy_loss, global_step)
        self.add_scalar("losses/entropy", entropy, global_step)
        self.add_scalar("losses/old_approx_kl", old_approx_kl, global_step)
        self.add_scalar("losses/approx_kl", approx_kl, global_step)
        self.add_scalar("losses/clipfrac", clipfrac, global_step)
        self.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - self.start_time)))
        self.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)        
