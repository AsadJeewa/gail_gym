import argparse
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from algo.behavior_clone import BehavioralCloning
from mlagents.envs import UnityEnvironment

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='name of directory to save model', default='trained_models/bc')
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/bc')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    return parser.parse_args()


def main(args):
    env = UnityEnvironment(file_name="C:/Users/ferra/py/Unity/RollerBall/RollerBall", worker_id=1, seed=0)
    train_mode = True 
    
    default_brain = env.brains[env.brain_names[0]]
    
    Policy = Policy_net('policy', default_brain)
    BC = BehavioralCloning(Policy)
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)

    observations = np.loadtxt('trajectory/observations.csv', dtype=None)#array

    #observations = observations.view().reshape(observations.shape + (-1,))
    actions = np.loadtxt('trajectory/actions.csv', dtype=np.int32)
    #print("OBV: ",len(observations[0]))
    #print("ACT: ",actions)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())
        
        inp = [observations, actions]#array of arrays
        #print("INP: ",inp," len: ",len(inp))
        for iteration in range(args.iteration):  # episode
            print("ITERATION: ",iteration," of ",args.iteration)
            # train
            #train minibatch epoch times, repeat number iterations
            for epoch in range(args.epoch_num):
                #print("EPOCH: ",epoch)
                #print("ERROR:",sess.run(BC.get_error()))
                # select sample indices in [low, high)
                #sample minibatch size random observations
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=args.minibatch_size)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                #print("OBV: ",sampled_inp[0].shape," | TYPE: ",type(sampled_inp[0])," | LEN: ",len(sampled_inp[0]))
                #print("ACT: ",sampled_inp[1].shape," | TYPE: ",type(sampled_inp[1])," | LEN: ",len(sampled_inp[1])," | CONTENT: ",sampled_inp[1])
                trainstep = BC.train(obs=sampled_inp[0], actions=sampled_inp[1])
                print("ERROR: ",trainstep[0])
            summary = BC.get_summary(obs=inp[0], actions=inp[1])

            if (iteration+1) % args.interval == 0:
                print("SAVE MODEL")
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)#apend step number

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
