import tensorflow as tf


class BehavioralCloning:
    def __init__(self, Policy):#, brain):
        self.Policy = Policy
        #self.brain = brain

        #act_space = brain.vector_action_space_size
        #print("BC ACT SPACE: ",act_space)
        self.actions_expert = tf.placeholder(tf.int32, shape=[None], name='actions_expert')
        self.act_dataset = tf.data.Dataset.from_tensor_slices(self.actions_expert)
        self.iter = self.act_dataset.make_initializable_iterator() # create the iterator    

        #shape = (?, numActions) hence depth is numActions: each action replaced with one hot vector e.g. 4 = (0,0,0,0,1)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!",self.Policy.act_probs)
        actions_vec = tf.one_hot(self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32)
        self.loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1)
        print("$$$",self.loss)
        self.loss = - tf.reduce_mean(self.loss)
        tf.summary.scalar('loss/cross_entropy', self.loss )

        optimizer = tf.train.AdamOptimizer()
        self.train_optimizer = optimizer.minimize(self.loss )

        self.merged = tf.summary.merge_all()

    def train(self, obs, actions):
        #print("BC obs: ",obs[0].shape," | ",self.Policy.obs.shape)
        #print("BC act: ",actions.shape," | ",self.actions_expert.shape)
        #optimiser, feed input, output
        #print(self.Policy.act_probs)
        #print(self.Policy.act_probs.shape[1])
        #print(actions_vec)
        return tf.get_default_session().run([self.loss,self.train_optimizer], feed_dict={self.Policy.obs: obs, self.actions_expert: actions})
        #return tf.get_default_session().run([self.train_optimizer,self.iter.initializer], feed_dict={self.Policy.obs: obs, self.actions_expert: actions})

    def get_summary(self, obs, actions):
        #return tf.get_default_session().run([self.merged,self.iter.initializer], feed_dict={self.Policy.obs: obs, self.actions_expert: actions})
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs, self.actions_expert: actions})

    #def get_error(self):
    #    return self.loss