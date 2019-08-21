import tensorflow as tf
import numpy as np

class Policy_net:
    def __init__(self, name: str, brain):
        """
        :param name: string
        :param env: unity env
        :param brain: unity brain
        """
        self.brain = brain
        ob_space = brain.vector_observation_space_size
        act_space = brain.vector_action_space_size

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape= [None,ob_space], name='obs')#shape is box shape shape= [ob_space]
            self.obs_dataset = tf.data.Dataset.from_tensor_slices(self.obs)
            self.iter = self.obs_dataset.make_initializable_iterator() # create the iterator


            #print("POLICY OBS: ",self.obs.shape)
            #given state (observations), output action i.e learn a policy
            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space[0], activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space[0], activation=tf.nn.softmax)
                #Softmax normalizes an input value into a vector of values that follows a probability distribution whose total sums up to 1.
                '''
                layer_1 = tf.keras.layers.Dense(units=20, input_shape=self.obs, activation=tf.tanh)
                layer_2 = tf.keras.layers.Dense(units=20, input_shape=layer_1, activation=tf.tanh)
                layer_3 = tf.layers.Dense(units=list(act_space)[0], input_shape=layer_2, activation=tf.tanh)
                self.act_probs = tf.layers.Dense(units=list(act_space)[0], input_shape=layer_3, activation=tf.nn.softmax)
                '''
            #given state, output value
            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)
                '''
                layer_1 = tf.keras.layers.Dense(units=20, input_shape=self.obs, activation=tf.tanh)
                layer_2 = tf.keras.layers.Dense(units=20, input_shape=layer_1, activation=tf.tanh)
                self.v_preds = tf.keras.layers.Dense(units=1, input_shape=layer_2, activation=None)
                '''
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)#max action

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            #return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
            return tf.get_default_session().run([self.act_stochastic, self.v_preds, self.iter.initializer], feed_dict={self.obs: obs})
            
        else:
            #return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})
            return tf.get_default_session().run([self.act_deterministic, self.v_preds,self.iter.initializer], feed_dict={self.obs: obs})
    def get_action_prob(self, obs):
        return tf.get_default_session().run([self.act_probs, self.iter.initializer], feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

