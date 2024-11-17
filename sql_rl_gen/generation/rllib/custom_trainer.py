import logging
import os
from pfrl.experiments.evaluator import Evaluator
from textrl import save_agent
from sql_rl_gen.generation.envs.utils import save_dict_csv

def train_evaulate_agent(agent, env, steps, eval_n_steps, eval_n_episodes, eval_interval, outdir, checkpoint_freq=None, train_max_episode_len=None, step_offset=0, eval_max_episode_len=None,
                         eval_env=None, successful_score=None, step_hooks=(), evaluation_hooks=(), save_best_so_far_agent=True, use_tensorboard=False, eval_during_episode=False, logger=None):
    for hook in evaluation_hooks:
        if not hook.support_train_agent:
            raise ValueError("{} does not support train_agent_with_evaluation().".format(hook))
    os.makedirs(outdir, exist_ok=True)
    if eval_env is None:
        assert not eval_during_episode, ("To run evaluation during training episodes, you need to specify `eval_env` that is independent from `env`.")
        eval_env = env
    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len
    evaluator = Evaluator(agent=agent, n_steps=eval_n_steps, n_episodes=eval_n_episodes, eval_interval=eval_interval, outdir=outdir, max_episode_len=eval_max_episode_len, env=eval_env,
                          step_offset=step_offset, evaluation_hooks=evaluation_hooks, save_best_so_far_agent=save_best_so_far_agent, use_tensorboard=use_tensorboard, logger=logger)
    eval_stats_history = train_agent(agent, env, steps, outdir, checkpoint_freq=checkpoint_freq, max_episode_len=train_max_episode_len, step_offset=step_offset, evaluator=evaluator,
                                     successful_score=successful_score, step_hooks=step_hooks, eval_during_episode=eval_during_episode, logger=logger)
    return agent, eval_stats_history

def train_agent(agent, env, steps, outdir, checkpoint_freq=None, max_episode_len=None, step_offset=0, evaluator=None, successful_score=None, step_hooks=(),
                eval_during_episode=False, logger=None, max_tokens=250):
    logger = logger or logging.getLogger(__name__)
    episode_r = 0
    episode_idx = 0
    obs = env.reset()
    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset
    eval_stats_history = []  # List of evaluation episode stats dict
    episode_len = 0
    tokens = 0
    try:
        while t < steps:
            action = agent.act(obs)
            obs, r, done, info = env.step(action)
            flag = True # Repeat the action if the reward is not 10.0
            attempts = 0
            max_attempts = 10  # Maximum number of attempts to repeat the action
            while flag:
                if done:
                    tokens = 0
                    reset = episode_len == max_episode_len or info.get("needs_reset", False)
                    agent.observe(obs, r, done, reset)
                    if r != 10.0 and attempts < max_attempts:
                        input = env.input_item
                        obs = env.reset(input)
                        action = agent.act(obs)
                        obs, r, done, info = env.step(action)
                        attempts += 1
                    else:
                        flag = False
                elif tokens >= max_tokens: # Penalise strictly if model generates a lot of stupid stuff
                    logger.info("Generated more tokens then allowed")
                    reset = episode_len == max_episode_len or info.get("needs_reset", False)
                    agent.observe(obs, -1000, done, reset)
                    obs = env.reset(env.input_item)
                    tokens = 0
                    action = agent.act(obs)
                    obs, r, done, info = env.step(action)
                    attempts += 1
                else:
                    action = agent.act(obs)  # Let the agent act again based on the new observation
                    obs, r, done, info = env.step(action)
                    tokens += 1
            t += 1
            episode_r += r
            episode_len += 1
            for hook in step_hooks:
                hook(env, agent, t)
            episode_end = done or reset or t == steps
            if episode_end:
                logger.info("outdir:%s step:%s episode:%s R:%s",outdir, t, episode_idx, episode_r)
                stats = agent.get_statistics()
                save_dict_csv(dict(stats), outdir, "scores")
                logger.info("statistics:%s", stats)
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
                episode_r = 0 # Start a new episode
                episode_len = 0
                obs = env.reset()
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")
    except (Exception, KeyboardInterrupt):
        save_agent(agent, t, outdir, logger, suffix="_except")
        raise
    save_agent(agent, t, outdir, logger, suffix="_finish")
    return eval_stats_history