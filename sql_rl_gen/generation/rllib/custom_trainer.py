import logging
import os
import time

import torch
from pfrl.experiments.evaluator import Evaluator
from textrl import save_agent
from tqdm import tqdm

from sql_rl_gen.generation.envs.utils import save_dict_csv

try:
    # tensorboardX is optional; training should not crash if it's unavailable or broken in the runtime.
    from tensorboardX import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def train_evaulate_agent(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    train_max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    eval_env=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    eval_during_episode=False,
    logger=None,
    log_interval=50,
    use_tqdm=True,
    print_every_step=False,
):
    for hook in evaluation_hooks:
        if not hook.support_train_agent:
            raise ValueError("{} does not support train_agent_with_evaluation().".format(hook))
    os.makedirs(outdir, exist_ok=True)
    if eval_env is None:
        assert not eval_during_episode, (
            "To run evaluation during training episodes, you need to specify `eval_env` that is independent from `env`."
        )
        eval_env = env
    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len
    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )
    eval_stats_history = train_agent(
        agent,
        env,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=train_max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        eval_during_episode=eval_during_episode,
        logger=logger,
        log_interval=log_interval,
        use_tqdm=use_tqdm,
        print_every_step=print_every_step,
    )
    return agent, eval_stats_history


def train_agent(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    eval_during_episode=False,
    logger=None,
    max_tokens=250,  # kept for backward compatibility
    log_interval=50,
    use_tqdm=True,
    print_every_step=False,
):
    logger = logger or logging.getLogger(__name__)
    episode_r = 0.0
    episode_idx = 0

    obs = env.reset()

    writer = None
    if SummaryWriter is not None:
        tb_logdir = os.path.join(outdir, "tb")
        try:
            writer = SummaryWriter(log_dir=tb_logdir)
        except Exception as e:
            logger.warning("TensorBoard logging disabled (SummaryWriter init failed): %s", e)
            writer = None
            try:
                if os.path.isdir(tb_logdir) and not os.listdir(tb_logdir):
                    os.rmdir(tb_logdir)
            except Exception:
                pass

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []
    episode_len = 0

    start_time = time.time()
    last_log_time = start_time
    last_logged_step = step_offset
    moving_reward_sum = 0.0
    moving_reward_count = 0

    pbar = tqdm(
        total=steps,
        initial=step_offset,
        disable=not use_tqdm,
        dynamic_ncols=True,
        desc="Training",
        mininterval=0.5,
    )

    try:
        while t < steps:
            action = agent.act(obs)
            obs, r, done, info = env.step(action)

            # NOTE: This environment is token-level (one token per step). The previous
            # implementation attempted to resample until a "perfect" reward, which biases
            # the training distribution and can lead to infinite retries; we strictly
            # advance by environment steps now.
            t += 1
            if use_tqdm:
                pbar.update(1)

            step_r = sum(r) if isinstance(r, (list, tuple)) else float(r)
            episode_r += step_r
            episode_len += 1

            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            for hook in step_hooks:
                hook(env, agent, t)

            episode_end = done or reset or t == steps

            if (t - last_logged_step) >= log_interval:
                now = time.time()
                elapsed = now - start_time
                window_elapsed = now - last_log_time if now > last_log_time else 0.0
                steps_per_sec_window = (
                    (t - last_logged_step) / window_elapsed if window_elapsed > 0 else 0.0
                )
                overall_steps_per_sec = (t - step_offset) / elapsed if elapsed > 0 else 0.0
                percent = 100.0 * (t / float(steps))
                gpu_mem = "NA"
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
                    gpu_mem = (
                        f"alloc={alloc:.1f}MB reserved={reserved:.1f}MB max_alloc={max_alloc:.1f}MB"
                    )
                avg_reward_so_far = (
                    (moving_reward_sum / moving_reward_count) if moving_reward_count > 0 else 0.0
                )
                logger.info(
                    "[progress] step %d/%d (%.1f%%) | eps %d | avgR %.3f | step/s(cur)=%.2f step/s(avg)=%.2f | elapsed %.1fs | gpu %s",
                    t,
                    steps,
                    percent,
                    episode_idx,
                    avg_reward_so_far,
                    steps_per_sec_window,
                    overall_steps_per_sec,
                    elapsed,
                    gpu_mem,
                )
                last_logged_step = t
                last_log_time = now

            if print_every_step:
                logger.info("[step] %d/%d r=%.4f eps=%d len=%d", t, steps, step_r, episode_idx, episode_len)

            if episode_end:
                logger.info("outdir:%s step:%s episode:%s R:%s", outdir, t, episode_idx, episode_r)
                if writer is not None:
                    writer.add_scalar("train/episode_reward", episode_r, t)

                stats = agent.get_statistics()
                save_dict_csv(dict(stats), outdir, "scores")
                logger.info("statistics:%s", stats)

                if use_tqdm:
                    pbar.set_postfix({"episode": episode_idx, "R": f"{episode_r:.3f}"})

                moving_reward_sum += episode_r
                moving_reward_count += 1
                episode_idx += 1

            if evaluator is not None and (episode_end or eval_during_episode):
                eval_score = evaluator.evaluate_if_necessary(t=t, episodes=episode_idx)
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                if successful_score is not None and evaluator.max_score >= successful_score:
                    break

            if episode_end:
                if t == steps:
                    break
                episode_r = 0.0
                episode_len = 0
                obs = env.reset()

            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        save_agent(agent, t, outdir, logger, suffix="_except")
        raise
    finally:
        if writer is not None:
            writer.close()
        if use_tqdm:
            pbar.close()

    save_agent(agent, t, outdir, logger, suffix="_finish")
    return eval_stats_history
