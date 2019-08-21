#!/usr/bin/python3
import argparse
from mlagents.envs import UnityEnvironment  
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
import time

#SCORE AND REWARD IS NOT THE SAME THING!!
#training restart game!??

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--max_success', default=5000, type=int)
    parser.add_argument('--discriminator_training_steps', default=10, type=int)
    parser.add_argument('--num_episodes_sample', default=10, type=int)#episodes * sample
    parser.add_argument('--interval', help='save interval', default=int(25), type=int)
    return parser.parse_args()

#SAMPLE, DISCRIMINATE, GENERATE
def main(args):
    f = open("results.txt","a")
    start_time = time.time()

    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBall/RollerBall", worker_id=1, seed=0)
    train_mode = True   
    brain_name = env.brain_names[0]
    default_brain = env.brains[brain_name]
    ob_space = default_brain.vector_observation_space_size
    Policy = Policy_net('policy', default_brain)
    Old_Policy = Policy_net('old_policy', default_brain)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    D = Discriminator(default_brain)

    #d_train = 0
    #p_train = 0
    cumulativeRewards = []

    expert_observations = np.loadtxt('trajectory/observations.csv', dtype=None)
    expert_actions = np.loadtxt('trajectory/actions.csv', dtype=np.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        env_info = env.reset(train_mode=train_mode)[brain_name]
        reward = 0  # do NOT use rewards to update policy
        scores = []
        success_num = 0

        for iteration in range(args.iteration):#repeat SAMPLE DISCRIM POLICY LOOP for N iterations 
            print("ITER ",iteration," of ",args.iteration," | SUCCESS ",success_num," of ",args.max_success)
            iteration_rewards = 0
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            #SAMPLE TRAJECTORIES
            for sample in range(args.num_episodes_sample):
                while True:
                    run_policy_steps += 1
                    obs = np.stack([env_info.vector_observations[0]]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                    act, v_pred, *_ = Policy.act(obs=obs, stochastic=True)#Policy_Net INPUT OBSERVATIONS
                    #policy iteratively improves
                    act = np.asscalar(act)
                    v_pred = np.asscalar(v_pred)

                    observations.append(obs)
                    actions.append(act)
                    rewards.append(reward)
                    v_preds.append(v_pred)

                    if(len(cumulativeRewards)>0):
                        #print(reward)
                        #print(cumulativeRewards[len(cumulativeRewards)-1])
                        cumulativeRewards.append(reward+cumulativeRewards[len(cumulativeRewards)-1])
                    else:
                        cumulativeRewards.append(reward)

                    env_info = env.step(act)[brain_name]
                    next_obs = env_info.vector_observations[0]
                    reward = env_info.rewards[0]
                    done = env_info.local_done[0]

                    iteration_rewards += reward

                    #if done and reward > 0:
                    if done:
                        v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                        env_info = env.reset(train_mode=train_mode)[brain_name]
                        break #only exit while loop if done (fell off board or reached goal)
                    else:
                        obs = next_obs
                #end while
                if(reward > 0):
                    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)]), success_num+1)
                #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)]), iteration)
                #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))]), iteration)

                if reward > 0:
                    success_num += 1
                    print("SUCCESS: ",success_num)
                    #render = True
                    if success_num >= args.max_success:
                        saver.save(sess, args.savedir + '/model.ckpt')
                        print('Clear!! Model saved.')
                        break#FINISH, STOP TRAINING

            scores.append(iteration_rewards)
            #print(scores)
            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + [ob_space])
            actions = np.array(actions).astype(dtype=np.int32)

            # train discriminator
            for i in range(args.discriminator_training_steps):#pass over n times
            #expert only used in Discriminator
                trainstep = D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)
            #d_train+=1
            # output of this discriminator is reward
            #d_rewards=[]
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            #print("length: ",len(d_rewards)," | ",np.mean(d_rewards))

            #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='discrim_rewards', simple_value=np.mean(d_rewards))]),d_train)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='discrim_loss', simple_value=np.mean(trainstep[1]))]),iteration)

            #print(d_rewards)
            #print("loss ",D.get_loss(agent_s=observations, agent_a=actions))
            #print("REWARDS TYPE RUN: ",type(d_rewards))
            #print("REWARDS SIZE RUN: ",len(d_rewards))

            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)#1D array
            ################################################################################
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)#value of each state
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # train policy
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            #print(observations.shape)#ndarray
            #print(actions.shape)#ndarray
            #print(gaes.shape)#ndarray
            #print(d_rewards.shape)#ndarray
            #print(v_preds_next.shape)#ndarray

            num_samples = observations.shape[0]
            num_iter = int(np.ceil(num_samples/args.minibatch_size))
            random_indices = np.arange(num_samples)
            np.random.shuffle(random_indices)#random shuffle does not return a result, it shuffles existing array 
            
            #print("samples: ",num_samples," | iter: ",num_iter)
            PPO.assign_policy_parameters()
            for train in range(num_iter):#train with n minibatches
                #randomly sample agent observations
                sample_indices = random_indices[train*args.minibatch_size:(train+1)*args.minibatch_size]
                #print(sample_indices)
                #sample_indices = np.random.randint(low=0, high=observations.shape[0],size=args.minibatch_size)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data

                trainstep = PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          gaes=sampled_inp[2],
                          rewards=sampled_inp[3],
                          v_preds_next=sampled_inp[4])

                print("TYPE: ", type(trainstep[1]), "|out ",trainstep[1])#, "|clip ",trainstep[2], "|vf ",trainstep[3], "|entropy ",trainstep[4])

            #p_train += 1
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='PPO_loss', simple_value=np.mean(trainstep[1]))]),iteration)
            #print("REWARDS TYPE RUN: ",type(d_rewards))
            #print("REWARDS SIZE RUN: ",len(d_rewards))
            
            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      gaes=inp[2],
                                      rewards=inp[3],
                                      v_preds_next=inp[4])
            ################################################################################
            writer.add_summary(summary, iteration)

            if (iteration+1) % args.interval == 0:
                print("SAVE MODEL")
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)#apend step number
        #end iteration
        writer.close()

        np.savetxt("gailTrainingScores.txt",scores)
        runtime = time.time() - start_time
        print("--- %s seconds ---" % runtime)
        f.write(str(runtime)+"\tGAIL\n")
        f.close()
        np.savetxt("GAILTrainingRewards.txt",cumulativeRewards)

if __name__ == '__main__':
    args = argparser()
    main(args)
