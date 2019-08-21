#!/usr/bin/python3
import argparse
from mlagents.envs import UnityEnvironment  
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    return parser.parse_args()


def main(args):
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBall/RollerBall", worker_id=1, seed=0)
    train_mode = True   
    brain_name = env.brain_names[0]
    default_brain = env.brains[brain_name]
    ob_space = default_brain.vector_observation_space_size
    Policy = Policy_net('policy', default_brain)
    Old_Policy = Policy_net('old_policy', default_brain)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    D = Discriminator(default_brain)

    expert_observations = np.loadtxt('trajectory/observations.csv', dtype=None)
    expert_actions = np.loadtxt('trajectory/actions.csv', dtype=np.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        env_info = env.reset(train_mode=train_mode)[brain_name]
        reward = 0  # do NOT use rewards to update policy
        success_num = 0

        for iteration in range(args.iteration):
            print("ITER ",iteration," of ",args.iteration)
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1
                obs = np.stack([env_info.vector_observations[0]]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                act, v_pred, *_ = Policy.act(obs=obs, stochastic=True)#Policy_Net INPUT OBSERVATIONS

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                env_info = env.step(act)[brain_name]
                next_obs = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    env_info = env.reset(train_mode=train_mode)[brain_name]
                    break #only exit ehile loop if done (fell off board or reached goal)
                else:
                    obs = next_obs
            #end while            
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if reward > 0:
                success_num += 1
                print("SUCCESS: ",success_num)
                #render = True
                if success_num >= 10:
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break#FINISH

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + [ob_space])
            actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator
            for i in range(2):
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
            ################################################################################
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()
            for epoch in range(6):
                #randomly sample agent observations
                sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                                   size=32)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])
            ################################################################################
            writer.add_summary(summary, iteration)
        #end iteration
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
