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
    parser.add_argument('--success_episodes', default=int(100))  
    parser.add_argument('--failure_episodes', default=int(100))
    parser.add_argument('--stochastic', action='store_false')
    return parser.parse_args()


def main(args):
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBall/RollerBall", worker_id=1, seed=1)
    
    train_mode = False
    random_agent = False

    f = open("testingResults.txt","a")

    success_steps=[]
    score = 0
    rewards = []
    cumulativeRewards = []
    num_iter=0
    #episode_rewards=[]
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
        failure_num = 0
        num_decisions = 0
        for iteration in range(args.iterations):
            print("ITERATION: ",(iteration+1)," of ",args.iterations)
            #rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                num_decisions += 1
                run_policy_steps += 1
                obs = np.stack([env_info.vector_observations[0]]).astype(dtype=np.float32)
                #print(obs.shape)
                if(random_agent):
                    act = np.column_stack([np.random.randint(0, default_brain.vector_action_space_size[i], size=(len(env_info.agents))) for i in range(len(default_brain.vector_action_space_size))])
                else:
                    act, *_ = Policy.act(obs=obs, stochastic=args.stochastic)

                #print("HELLO", act)
                act = np.asscalar(act)

                env_info = env.step(act)[brain_name]
                next_obs = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                if(len(cumulativeRewards)>0):
                    cumulativeRewards.append(reward+cumulativeRewards[len(cumulativeRewards)-1])
                else:
                    cumulativeRewards.append(reward)
                rewards.append(reward)

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
            '''
            if reward < 0:
                print("GAME OVER!")
                cumulativeRewards.append(reward+cumulativeRewards[len(cumulativeRewards)-1])
                #print(success_num," successes in ",iteration+1," iterations")
                print("Your score:",score,"(",num_decisions,"steps )")
                break
            elif reward > 0:
            '''
            if reward > 0:
                success_num += 1
                score += 1
                print("Success ",success_num," of ",args.success_episodes)
                print("episode steps till success: ",run_policy_steps)
                success_steps.append(run_policy_steps)

            if reward < 0:
                failure_num += 1
                print("Failure ",failure_num)
                print("episode steps till failure: ",run_policy_steps)

            print("\n")
                
            if success_num >= args.success_episodes or failure_num >= args.failure_episodes:
                print("GAME OVER!")
                num_iter = iteration+1
                cumulativeRewards.append(reward+cumulativeRewards[len(cumulativeRewards)-1])
                rewards.append(reward)
                print("Your score:",score,"(",num_decisions,"steps )")
                break
                
        writer.close()
        avg_steps = np.mean(np.array(success_steps))
        print("ITERATIONS: ",num_iter,"\nSUCCESSES: ", success_num,"\nFAILURES: ",failure_num,"\nAVERAGE SUCCESS STEPS: ",avg_steps)
        if(not random_agent):
            f.write("\n"+str(args.alg)+"\nITERATIONS: "+str(num_iter)+"\nSUCCESSES: "+str(success_num)+"\nFAILURES: "+str(failure_num)+"\nAVERAGE SUCCESS STEPS: "+str(avg_steps))
        else:
            f.write("\nRANDOM\nITERATIONS: "+str(num_iter)+"\nSUCCESSES: "+str(success_num)+"\nFAILURES: "+str(failure_num)+"\nAVERAGE SUCCESS STEPS: "+str(avg_steps))
        f.close()
    np.savetxt("rewards.txt",cumulativeRewards)

if __name__ == '__main__':
    args = argparser()
    main(args)
