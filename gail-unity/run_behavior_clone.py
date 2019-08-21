import argparse
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.behavior_clone import BehavioralCloning
from mlagents.envs import UnityEnvironment
import time

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/bc')
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/bc')
    #parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(25), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=250, type=int)
    return parser.parse_args()


def main(args):
    f = open("results.txt","a")
    start_time = time.time()
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBallExperimental/RollerBall", worker_id=1, seed=0)
    train_mode = True 
    
    default_brain = env.brains[env.brain_names[0]]
    
    Policy = Policy_net('policy', default_brain)
    BC = BehavioralCloning(Policy)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    observations = np.loadtxt('trajectory/observations.csv', dtype=None)#array

    #observations = observations.view().reshape(observations.shape + (-1,))
    actions = np.loadtxt('trajectory/actions.csv', dtype=np.int32)
    
    num_samples = observations.shape[0]
    num_iter = int(np.ceil(num_samples/args.minibatch_size))
    #print(num_samples," | ",num_iter)
    error = 0
    #print("OBV: ",len(observations[0]))
    #print("ACT: ",actions)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        inp = [observations, actions]#array of arrays
        #print("INP: ",inp," len: ",len(inp))
        for epoch in range(args.epoch_num):#pass over entire training data for N epochs
            print("EPOCH: ",(epoch+1)," of ",args.epoch_num)
            # train
            random_indices = np.arange(num_samples)
            #print (random_indices)
            np.random.shuffle(random_indices)#random shuffle does not return a result, it shuffles existing array                
            #print (random_indices)
            #print ("LEN: ",len(random_indices))
            #print (random_indices[:args.minibatch_size])
            for iteration in range(num_iter):
                #print("ITERATION: ",(iteration+1)," of ",num_iter)
                #print("ERROR:",sess.run(BC.get_error()))
                # select sample indices in [low, high)
                #sample minibatch size random observations
                #sample_indices = np.random.randint(low=0, high=observations.shape[0], size=args.minibatch_size)
                #sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                #print("OBV: ",sampled_inp[0].shape," | TYPE: ",type(sampled_inp[0])," | LEN: ",len(sampled_inp[0]))
                #print("ACT: ",sampled_inp[1].shape," | TYPE: ",type(sampled_inp[1])," | LEN: ",len(sampled_inp[1])," | CONTENT: ",sampled_inp[1])
                
                #print(iteration*args.minibatch_size," | YELLO |",(iteration+1)*args.minibatch_size)
                sample_indices = random_indices[iteration*args.minibatch_size:(iteration+1)*args.minibatch_size]
                #sample_indices = random_indices[128:256]
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                #print("@@@@@@@@",sampled_inp[1].shape)
                #print(sampled_inp[1])
                #print(sample_indices, " | INDICES |")
                #print(sampled_inp," | SELECTED |")
                trainstep = BC.train(obs=sampled_inp[0], actions=sampled_inp[1])
                error = trainstep[0]
                #print("ERROR: ",trainstep[0])
            print("ERROR: ",error)
            summary = BC.get_summary(obs=inp[0], actions=inp[1])
            #print("\n")
            if (epoch+1) % args.interval == 0:
                print("SAVE MODEL")
                saver.save(sess, args.savedir + '/model.ckpt', global_step=epoch+1)#apend step number

            if(error<0.1):
                print("STOP TRAINING")
                print("SAVE MODEL")
                saver.save(sess, args.savedir + '/model.ckpt', global_step=epoch+1)#apend step number
                break;
            writer.add_summary(summary, iteration)
        writer.close()
        runtime = time.time() - start_time
        print("--- %s seconds ---" % runtime)
        f.write(str(runtime)+"\tBC\n")
        f.close()

if __name__ == '__main__':
    args = argparser()
    main(args)
