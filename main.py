# Third party imports
import os

# Local module imports
from agent_class import Agent
from env_gym import BipedEnv


gym_env = BipedEnv()
agent = Agent(gym_env)
score_tracker = []


def setup_log_files():
    if not agent.testing:
        if agent.delete_log:
            if os.path.exists("log_file.csv"):
                print("Deleting log file")
                os.remove("log_file.csv")

        if agent.load_log:
            agent.load_info_log()

        if agent.load_model_status:
            agent.load_models()

    if agent.testing:
        agent.load_models()


def training_evaluation_function():
    for _ in range(1, agent.episodes+1):
        state = agent.env.reset_env()

        # Trackers
        done = False
        score = 0
        time_steps = 0
        learning_iterations = 0
        agent.episode_count += 1

        while not done:
            action = agent.do_action(state)
            next_state, reward, done, _ = agent.env.env_action(action)

            agent.to_memory(state, action, reward, next_state, done)

            agent.train()
            learning_iterations += 1

            state = next_state
            time_steps += 1
            score += reward

            if done:
                score_tracker.append(score)

                if len(score_tracker) < 10:
                    ave_score = 0
                else:
                    ave_score = sum(score_tracker[-10:]) / 10
                    del (score_tracker[:-10])

                print(
                    f"Episode: {agent.episode_count} - "
                    f"Episode Score: {round(score, 0)} - "
                    f"10 Round Ave: {round(ave_score, 0)}")

                if agent.save_log:
                    Agent.log_info(agent.episode_count, round(score, 0), agent.tau)

        if agent.episode_count % agent.episodes_to_save == 0 and agent.save_model_status is True:
            agent.save_models()


if __name__ == "__main__":
    setup_log_files()
    training_evaluation_function()
