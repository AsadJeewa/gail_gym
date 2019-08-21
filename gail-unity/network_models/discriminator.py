import tensorflow as tf


class Discriminator:
    def __init__(self, brain):
        """
        :param brain:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).

        """

        self.ob_space = brain.vector_observation_space_size #state
        self.act_space = brain.vector_action_space_size  

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + [self.ob_space])
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=self.act_space[0])
            # add noise for stabilise training
            #random values from a normal distribution
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)

            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + [self.ob_space])
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=self.act_space[0])
            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a)#prob real image is real
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=agent_s_a)#prob fake image is real
                #print("##################################",tf.shape(prob_2))#tensor

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            self.loss_out = loss
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)
            #self.lossOut = loss
            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent
            #agent wants to fool discriminator so prob_2 must be maximised
            #print("REWARDS: ",type(self.rewards))

    def construct_network(self, input):
        #print("############################### NETWORK CONSTRUCT: ",input.get_shape)
        #print("############################### OBS ACT SPACE: ",self.ob_space," | ",self.act_space)
        layer_1 = tf.layers.dense(inputs=input, units=20, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name='layer3')
        #Leaky ReLUs allow a small, positive gradient when the unit is not active
        prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.nn.sigmoid, name='prob')
        #print("##################################",self.prob.get_shape()," | ", type(self.prob))#tensor
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run([self.train_op,self.loss_out], feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        #print("PROB TYPE IN CLASS: ",type(self.prob))
        #print("PROB SHAPE IN CLASS: ",self.prob.get_shape())
        #print("REWARDS TYPE IN CLASS: ",type(self.rewards))
        #print("REWARDS SHAPE IN CLASS: ",self.rewards.get_shape())
        
        #rew = tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
        #                                                             self.agent_a: agent_a})
        #print("AGENTS SHAPE IN CLASS2: ",agent_s.shape)
        #print("AGENTS SHAPE IN CLASS2: ",agent_a.shape)

        #print("PROB_2 SHAPE IN CLASS2: ",prob_2.shape)
        #print("REWARDS TYPE IN CLASS2: ",type(rew))
        #print("REWARDS SHAPE IN CLASS2: ",len(rew))
        #return rew
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

'''
    def get_loss(self):
        return tf.get_default_session().run(self.lossOut, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})
'''