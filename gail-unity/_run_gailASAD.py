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
    #parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--interval', help='save interval', default=int(25), type=int)
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--epoch_num', default=250, type=int)
    return parser.parse_args()


def main(args):
    #GENERATOR
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBall/RollerBall", worker_id=1, seed=5)
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

        num_samples = expert_observations.shape[0]
        num_iter = int(np.ceil(num_samples/args.minibatch_size))
        for epoch in range(args.epoch_num):
            print("EPOCH: ",(epoch+1)," of ",args.epoch_num)
        #for iteration in range(args.iterations):
            #print("ITER ",iteration," of ",args.iteration)
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            
            #single episode
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
                    print("DONESO")
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    env_info = env.reset(train_mode=train_mode)[brain_name]
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)]))
                              # , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]))
                              # , iteration)
            '''                  
            if reward > 0:
                success_num += 1
                print("run steps: ",run_policy_steps)
                if success_num >= args.episodes:
                    print(success_num," successes in ",iteration+1," iterations")
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            #else:
            #    success_num = 0
        '''
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

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            PPO.assign_policy_parameters()

            random_indices = np.arange(num_samples)
            np.random.shuffle(random_indices)   
            print("num_iter: ",num_iter)
            for iteration in range(num_iter):
                sample_indices = random_indices[iteration*args.minibatch_size:(iteration+1)*args.minibatch_size]
                print(sample_indices)
                print("INP: ",inp[0].shape)
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
            if (epoch+1) % args.interval == 0:
                print("SAVE MODEL")
                saver.save(sess, args.savedir + '/model.ckpt', global_step=epoch+1)#apend step number
            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
