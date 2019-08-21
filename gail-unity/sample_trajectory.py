import sys
import argparse
from mlagents.envs import UnityEnvironment
import numpy as np
from network_models.policy_net import Policy_net
import tensorflow as tf


# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='filename of model to test', default='trained_models/ppo/model.ckpt')
    parser.add_argument('--iterations', default=20, type=int)

    return parser.parse_args()


def main(args):
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/BASICNeg/Unity Environment", worker_id=1, seed=0)
    train_mode = False 
    # Set the default brain to work with
    brain_name = env.brain_names[0]
    default_brain = env.brains[brain_name]

    env_info = env.reset(train_mode=train_mode)[brain_name]
    
    # Examine the state space for the default brain
    ob_space = default_brain.vector_observation_space_size
    
    
    Policy = Policy_net('policy', default_brain)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, args.model) #IMP
        env_info = env.reset(train_mode=train_mode)[brain_name]
        #print("DONE OUT: ",env_info.local_done)
        for iteration in range(args.iterations):  # episode
            print("ITER:",iteration)
            observations = []
            actions = []
            rewards= []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                #obs = np.stack(env_info.vector_observations[0]).astype(dtype=np.float32)
                #obs = [env_info.vector_observations[0]]#.astype(dtype=np.float32)
                obs = np.stack([env_info.vector_observations[0]]).astype(dtype=np.float32)
                #print(obs.shape)
                act, *_ = Policy.act(obs=obs, stochastic=True)
                #print("ACTION: ",act)
                act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)

                #next_obs, reward, done, info = env.step(act)
                env_info = env.step(act)[brain_name]
                next_obs = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                #print("DONE POST2: ",env_info.local_done)
                #print(reward)
                if(reward>0):
	                print("REWARD!! ",reward)
                #elif(reward<0):
                #   print("PUNISH!!")   
                if done:
                    print("DONE!!!!!!!!!!")
                    print(run_steps)
                    obs = env.reset(train_mode=train_mode)[brain_name]
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + [ob_space])
            actions = np.array(actions).astype(dtype=np.int32)

            open_file_and_save('trajectory/observationsPPO.csv', observations)
            open_file_and_save('trajectory/actionsPPO.csv', actions)


if __name__ == '__main__':
    args = argparser()
    main(args)
