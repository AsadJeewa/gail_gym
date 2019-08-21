#!/usr/bin/python3
import argparse
from mlagents.envs import UnityEnvironment
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iterations', default=int(1e5), type=int)
    parser.add_argument('--stochastic', action='store_false')
    parser.add_argument('--minibatch_size', default=32, type=int)
    return parser.parse_args()


def main(args):
    #env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBall/RollerBall", worker_id=1, seed=5)
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/BASICNeg/Unity Environment", worker_id=1, seed=0)
    
    train_mode = True 
    brain_name = env.brain_names[0]
    default_brain = env.brains[brain_name]
    #env.seed(0)
    ob_space = default_brain.vector_observation_space_size
    Policy = Policy_net('policy', default_brain)
    Old_Policy = Policy_net('old_policy', default_brain)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        env_info = env.reset(train_mode=train_mode)[brain_name]
       
        reward = 0
        success_num = 0
        #render = False
        for iteration in range(args.iterations):
            print("iteration ",iteration," of ",args.iterations)
            observations = []
            actions = []
            v_preds = []
            rewards = []
            episode_length = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                episode_length += 1
                obs = np.stack([env_info.vector_observations[0]]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred, *_ = Policy.act(obs=obs, stochastic=args.stochastic)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                env_info = env.step(act)[brain_name]
                next_obs = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                #if render:
                #	env.render()
                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    env_info = env.reset(train_mode=train_mode)[brain_name]
                    #print("DONE")
                    #print(len(observations))
                    #reward = -1 #WHY!?
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            #print("THIS REWARD: ",reward)
            if reward >= 0.9:#FIX 100 Roller Ball, 1 for Basic (Rounding error)
                print("Goal: ",episode_length)
                print("iteration ",iteration," of ",args.iterations)
                if(episode_length == 7):
                    print("******REWARDED: ",success_num)
                    success_num += 1
                #render = True
                if success_num >= 100:#consecutive successes
                    saver.save(sess, args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            #print(len(observations))
            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + [ob_space])
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]
            '''
            num_samples = observations.shape[0]
            num_iter = int(np.ceil(num_samples/args.minibatch_size))
            random_indices = np.arange(num_samples)
            np.random.shuffle(random_indices)#random shuffle does not return a result, it shuffles existing array
            '''
            # train
            #for train in range(num_iter):
            for epoch in range(6):
                # sample indices from [low, high)
                
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                #sample_indices = random_indices[train*args.minibatch_size:(train+1)*args.minibatch_size]
                #print(sample_indices)
                
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

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
