from mlagents.envs import UnityEnvironment
import numpy as np
import tensorflow as tf
import argparse
from network_models.policy_net import Policy_net


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='directory of model', default='trained_models')
    parser.add_argument('--alg', help='choose algorithm one of gail, ppo, bc', default='bc')
    parser.add_argument('--model', help='number of model to test. model.ckpt-number', default='250')
    parser.add_argument('--logdir', help='log directory', default='log/test')
    parser.add_argument('--iterations', default=int(1000))
    parser.add_argument('--episodes', default=int(10))
    parser.add_argument('--stochastic', action='store_false')
    return parser.parse_args()


def main(args):
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBallExperimental/RollerBall", worker_id=1, seed=5)
    train_mode = False
    
    score = 0
    rewards = []
    cumulativeRewards = []
    brain_name = env.brain_names[0]     
    default_brain = env.brains[brain_name]   
    env_info = env.reset(train_mode=train_mode)[brain_name]  

    ob_space = default_brain.vector_observation_space_size
    
    Policy = Policy_net('policy', default_brain)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir+'/'+args.alg, sess.graph)
        sess.run(tf.global_variables_initializer())
        if args.model == '':
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+'model.ckpt')
        else:
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+'model.ckpt-'+args.model)
        env_info = env.reset(train_mode=train_mode)[brain_name]
        reward = 0
        success_num = 0
        num_decisions = 0
        for iteration in range(args.iterations):
            #rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                num_decisions += 1
                run_policy_steps += 1
                obs = np.stack([env_info.vector_observations[0]]).astype(dtype=np.float32)
                act, *_ = Policy.act(obs=obs, stochastic=args.stochastic)

                act = np.asscalar(act)

                if(len(cumulativeRewards)>0):
                    cumulativeRewards.append(reward+cumulativeRewards[len(cumulativeRewards)-1])
                else:
                    cumulativeRewards
                    cumulativeRewards.append(reward)
                rewards.append(reward)

                env_info = env.step(act)[brain_name]
                next_obs = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                #scores.append(score)
                #print("STEP ",num_decisions,": ", score)
                #env.render()
                if done:
                    env_info = env.reset(train_mode=train_mode)[brain_name]
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            # end condition of test
            if reward < 0:
                print("GAME OVER!")
                cumulativeRewards.append(reward+cumulativeRewards[len(cumulativeRewards)-1])
                #print(success_num," successes in ",iteration+1," iterations")
                print("Your score:",score,"(",num_decisions,"steps )")
                break
            elif reward > 0:
                success_num += 1
                score += 1
                print("episode steps till success: ",run_policy_steps)
                '''
                if success_num >= args.episodes:
                    print(success_num," successes in ",iteration+1," iterations")
                    print('DONE!!')
                    break
                '''
        writer.close()
    np.savetxt("rewards.txt",cumulativeRewards)

if __name__ == '__main__':
    args = argparser()
    main(args)
