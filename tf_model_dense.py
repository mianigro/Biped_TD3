import tensorflow as tf
import tensorflow_probability as tfp
import keras

# Q of state action pair
class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=256, fc3_dims=128):
        super(CriticNetwork, self).__init__()

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.fc3 = tf.keras.layers.Dense(fc3_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = self.fc3(q1_action_value)

        q = self.q(q1_action_value)

        return q


# Policy
class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=256, fc3_dims=128):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.fc3 = tf.keras.layers.Dense(fc3_dims, activation='relu')

        # Multiple mu by max actions for environments above (-1,1)
        self.mu = tf.keras.layers.Dense(self.n_actions, activation="tanh")

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)

        mu = self.mu(prob)

        return mu