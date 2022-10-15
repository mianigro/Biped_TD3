# Python imports
import os
import shutil
from csv import writer, reader

# Third party imports
import numpy as np
import tensorflow as tf
import keras

# Module imports
from replay_buffer import ReplayBuffer
from tf_model_dense import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, environment):
        # Environment setup from the input environment
        self.env = environment
        self.state_size = self.env.state_size
        self.n_actions = self.env.n_actions
        self.max_action = self.env.max
        self.min_action = self.env.min

        # Hyperparameters
        self.mem_size = 100000
        self.gamma = 0.99
        self.tau = 0.005
        self.noise = 0.1  # SD for extra noise for exploration/exploitation
        self.scale = 0.2  # SD for extra noise for target action of new state (critic loss)
        self.update_actor_iter = 2
        self.warmup = 3000

        # Counters
        self.episode_count = 0
        self.learn_step_counter = 0
        self.time_step = 0

        # Training hyperparameters
        self.episodes = 10000
        self.batch_size = 100
        self.learning_rate_critic = 0.001
        self.learning_rate_actor = 0.0005
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_actor)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_critic)

        # Data saving
        self.testing = False
        self.save_log = True
        self.load_log = False
        self.delete_log = True
        self.save_model_status = True
        self.load_model_status = False
        self.episodes_to_save = 10

        # Memory
        self.memory = ReplayBuffer(
            self.mem_size,
            self.state_size,
            self.n_actions,
            self.batch_size)

        # Build models
        self.actor = ActorNetwork(self.n_actions)
        self.critic_1 = CriticNetwork()
        self.critic_2 = CriticNetwork()
        self.target_actor = ActorNetwork(self.n_actions)
        self.target_critic_1 = CriticNetwork()
        self.target_critic_2 = CriticNetwork()

        # Compile models
        self.actor.compile(optimizer=self.a_opt, loss="mean")
        self.critic_1.compile(optimizer=self.c_opt, loss="mean_squared_error")
        self.critic_2.compile(optimizer=self.c_opt, loss="mean_squared_error")
        self.target_actor.compile(optimizer=self.a_opt, loss="mean")
        self.target_critic_1.compile(optimizer=self.c_opt, loss="mean_squared_error")
        self.target_critic_2.compile(optimizer=self.c_opt, loss="mean_squared_error")

        # Update target_value weights from value network
        self.update_network_parameters(tau=1)

    def to_memory(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)

    def do_action(self, state):
        # Warmup
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions, ))
        else:
            state = tf.convert_to_tensor([state], dtype=tf.float32)  # Putting [] adds batch dimension
            mu = self.actor(state)[0]

        # Add noise and clip
        mu_prime = mu + np.random.normal(scale=self.noise)
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        self.time_step += 1

        return mu_prime

    # Update weights from value networks to target networks modified by tau
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_critic_2.set_weights(weights)

    def train(self):
        if self.memory.memory_count < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Train critic
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_states)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=self.scale), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            # New states
            q1_new = self.target_critic_1(next_states, target_actions)
            q2_new = self.target_critic_2(next_states, target_actions)
            q1_new = tf.squeeze(q1_new, 1)
            q2_new = tf.squeeze(q2_new, 1)

            # Current states
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            # Critic values to find target
            critic_value_next = tf.math.minimum(q1_new, q2_new)
            target = rewards + self.gamma * critic_value_next * (1 - done)

            # Losses of critics
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        # Update gradients
        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_iter != 0:
            return

        # Train actor
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    @staticmethod
    def log_info(episode, score, tau):
        new_row = [episode, score, tau]

        with open('log_file.csv', 'a') as log_file:
            writer_object = writer(log_file)
            writer_object.writerow(new_row)

    def load_info_log(self):
        if os.path.exists("log_file.csv"):
            with open('log_file.csv', 'r') as log_file:
                data = list(reader(log_file, delimiter=","))

            episode_count, score, tau = data[-1]
            self.episode_count = int(episode_count)
            self.tau = float(tau)
            self.update_network_parameters()
            print("Log file loaded.")
        else:
            print("No log file to load.")

    def save_models(self):
        if os.path.isdir("actor"):
            shutil.rmtree("actor")
        if os.path.isdir("critic_1"):
            shutil.rmtree("critic_1")
        if os.path.isdir("critic_2"):
            shutil.rmtree("critic_2")
        if os.path.isdir("target_actor"):
            shutil.rmtree("target_actor")
        if os.path.isdir("target_critic_1"):
            shutil.rmtree("target_critic_1")
        if os.path.isdir("target_critic_2"):
            shutil.rmtree("target_critic_2")

        self.actor.save_weights(f"actor/actor")
        self.critic_1.save_weights(f"critic_1/critic_1")
        self.critic_2.save_weights(f"critic_2/critic_2")
        self.target_actor.save_weights(f"target_actor/target_actor")
        self.target_critic_1.save_weights(f"target_critic_1/target_critic_1")
        self.target_critic_2.save_weights(f"target_critic_2/target_critic_2")

        print("Models saved.")

    def load_models(self):
        if os.path.isdir("actor") and \
                os.path.isdir("critic_1") and \
                os.path.isdir("critic_2") and \
                os.path.isdir("target_actor") and \
                os.path.isdir("target_critic_1") and \
                os.path.isdir("target_critic_2"):

            self.actor.load_weights("actor/actor")
            self.critic_1.load_weights("critic_1/critic_1")
            self.critic_2.load_weights("critic_2/critic_2")
            self.target_actor.load_weights("target_actor/target_actor")
            self.target_critic_1.load_weights("target_critic_1/target_critic_1")
            self.target_critic_2.load_weights("target_critic_2/target_critic_2")
            self.update_network_parameters(tau=1)
            print("Previous weights loaded.")

        else:
            print("No model to load")


