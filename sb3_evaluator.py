import os
import sys
from chrono import Chrono
import argparse
import numpy as np
from itertools import count
from utils import ALGOS, create_test_env, get_saved_hyperparams
from stable_baselines3.common.evaluation import evaluate_policy


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    args = parser.parse_args()
    return args


def read_name(filename):
    """
    :param filename: the file name, including the path
    :return: fields
    """
    fields = filename.split('#')
    tmp = fields[0]
    env_name = tmp.split('/')
    env_name = env_name[-1]
    algo = fields[1]
    end_name = fields[2]
    team_name = end_name.split('.')
    return env_name, algo, team_name[0]


def get_scores(args, folder, policy_file, env_name, algo, stats_path, hyperparams, n_evals):
    """
    """
    env = create_test_env(
        env_name,
        n_envs=10,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=folder + "/../Logs",
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs={},
    )
    fields = policy_file.split('.')


    custom_objects = {}
    # This part is useful if the generated files come from a more recent version of python
    # newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
    # if newer_python_version:
    # custom_objects = {
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "exploration_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     }

    model = ALGOS[algo].load(folder + "/" + fields[0], custom_objects=custom_objects)
    policy = model.policy
    episode_rewards, _ = evaluate_policy(policy, env, n_eval_episodes=n_evals, return_episode_rewards=True)
    scores = np.array(episode_rewards)
    return scores


class Evaluator:
    """
    A class to evaluate a set of policies stored into the same folder and ranking them according to their scores
    """
    def __init__(self):
        self.score_dic = {}

    def load_policies(self, folder) -> None:
        """
         :param: folder : name of the folder containing policies
         Output : none (policies of the folder stored in self.env_dict)        
         """
        listdir = os.listdir(folder)
        args = parse()
        stats_path = folder
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)
        n_evals = 200

        for policy_file in listdir:
            print(policy_file)
            env_name, algo, team_name = read_name(policy_file)

            if env_name in self.score_dic:
                scores = get_scores(args, folder, policy_file, env_name, algo, stats_path, hyperparams, n_evals)
                self.score_dic[env_name][scores.mean()] = [team_name, algo, scores.std()]
            else:
                scores = get_scores(args, folder, policy_file, env_name, algo, stats_path, hyperparams, n_evals)
                tmp_dic = {scores.mean(): [team_name, algo, scores.std()]}
                self.score_dic[env_name] = tmp_dic

    def display_hall_of_fame(self) -> None:
        """
        Display the hall of fame of all the evaluated policies
        :return: nothing
        """
        print("Hall of fame")
        for env, dico in self.score_dic.items():
            print("Environment :", env)
            for key, val in sorted(dico.items(),reverse=True):
                print("team: ", val[0], "           algo:", val[1], "      mean score: ", key, "std: ", val[2])


if __name__ == '__main__':
    directory = os.getcwd() + '/data/policies/'
    ev = Evaluator()
    c = Chrono()
    ev.load_policies(directory)
    ev.display_hall_of_fame()
    c.stop()
