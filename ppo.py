import random
import numpy as np
import torch
import env
import iagent
import summary


def train(args, run_name):

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")

    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    writer = summary.Writer(args, run_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("current device: cuda" if torch.cuda.is_available()
          and args.cuda else "current device: cpu")
    envs = env.Enviroment(args=args, run_name=run_name)

    agent = iagent.Agent(args, envs, device, run_name)
    if args.load_model is not None:
        agent.load_model(args.load_model)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        """ agent.reset_actor_critic_states() """
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            agent.actor.optimizer.param_groups[0]["lr"] = lrnow
            agent.critic.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy())
            done = np.logical_or(truncated, terminated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(done).to(device)

            for key in info:
                if key == "final_info":
                    for item in info[key]:
                        if item != None:
                            for in_key in item.keys():
                                if in_key == 'episode':
                                    writer.episode_summary(global_step,

                                                           episodic_return=float(
                                                               item['episode']['r']),
                                                           episodic_length=float(item["episode"]["l"]))
                                    break

        # bootstrap value if not done
        with torch.no_grad():
            value = agent.get_value(next_obs)
            next_value = value.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * \
                        nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * \
                        args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * \
                        nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds])
                newvalue = value
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Optimization
                agent.backprop(loss)

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.final_summary(global_step,
                             learning_rate=agent.actor.optimizer.param_groups[0]["lr"],
                             value_loss=v_loss.item(),
                             policy_loss=pg_loss.item(),
                             entropy=entropy_loss.item(),
                             old_approx_kl=old_approx_kl.item(),
                             approx_kl=approx_kl.item(),
                             clipfrac=np.mean(clipfracs),
                             explained_var=explained_var
                             )

        if args.save_on_train:
            agent.save_model()

    if args.save_on_exit:
        agent.save_model()

    envs.close()
    writer.close()


def test(args, run_name):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    tenv = env.make_test_env(gym_id=args.gym_id, seed=args.seed)

    writer = summary.Writer(args, run_name)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    tagent = iagent.TestAgent(args, tenv, device, run_name)

    if args.load_model is not None:
        tagent.load_model(args.load_model)
    else:
        tagent.load_current_model()
    obs, _ = tenv.reset(seed=args.seed)
    done = False
    while True:
        if done == True:
            obs, _ = tenv.reset()
            """ tagent.reset_actor_critic_states() """
            done = False
            for key in info:
                if key == "final_info":
                    for item in info[key]:
                        if item != None:
                            for in_key in item.keys():
                                if in_key == 'episode':
                                    writer.test_summary(
                                        episodic_return=float(
                                            item['episode']['r'])
                                    )
                                    break
        # Random action
        obs = torch.Tensor(obs).to(device)
        action = tagent.get_action(obs)
        obs, _, terminated, truncated, info = tenv.step(
            action=int(action.flatten().cpu().detach().numpy()))
        done = np.logical_or(truncated, terminated)
