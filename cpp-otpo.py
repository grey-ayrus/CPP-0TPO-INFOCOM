import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(x, dim=-1)

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class DRL_CPP:
    def __init__(self, actor_input_size, actor_output_size, critic_input_size, critic_output_size, alpha, beta, epsilon, epsilon_0, beta_time, T):
        self.actor = Actor(actor_input_size, actor_output_size)
        self.critic = Critic(critic_input_size, critic_output_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.epsilon_0 = epsilon_0
        self.beta_time = beta_time
        self.T = T
        self.exploration_duration = 0
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=alpha)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=beta)

    def explore_exploit(self, states):
        actions = []
        for state in states:
            if np.random.rand() < self.epsilon:
                action = np.random.rand(2)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_probs = self.actor(state_tensor).squeeze().numpy()
                    action = np.random.choice(range(len(action_probs)), p=action_probs)
            actions.append(action)
        return actions

    def update_epsilon(self, t):
        self.epsilon = self.epsilon_0 * (1 - self.beta_time) ** t

    def update_actor_critic(self, states, actions, rewards, next_states):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        with torch.no_grad():
            target_value = rewards_tensor + self.critic(next_states_tensor)
        
        current_value = self.critic(states_tensor)
        td_error = target_value - current_value
        
        log_probs = torch.log(self.actor(states_tensor)[range(len(actions)), actions])
        actor_loss = -log_probs * td_error
        critic_loss = 0.5 * td_error ** 2
        
        self.optimizer_actor.zero_grad()
        actor_loss.mean().backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.mean().backward()
        self.optimizer_critic.step()

class OTPO:
    def __init__(self, alpha, batch_size, num_agents, episodes, steps_per_episode, L):
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.L = L
        self.agents = [Agent() for _ in range(num_agents)]

    def explore_exploit(self, epsilon):
        if np.random.rand() < epsilon:
            return "Explore"
        else:
            return "Exploit"

    def adaptive_exploration_probability(self, epsilon, t):
        return epsilon * (1 - self.alpha) ** t

    def update_actor(self, agent, gradient):
        self.agents[agent].update_network(gradient)

    def run(self):
        for k in range(self.episodes):
            for t in range(self.steps_per_episode):
                epsilon = self.adaptive_exploration_probability(self.alpha, t)
                for agent in range(self.num_agents):
                    action = self.explore_exploit(epsilon)
                    if action == "Explore":
                        self.agents[agent].exploration_duration += 1
                        if self.agents[agent].exploration_duration >= self.agents[agent].T:
                            self.agents[agent].update_epsilon(t)
                            self.agents[agent].exploration_duration = 0
                    else:
                        self.agents[agent].exploration_duration = 0

class Agent:
    def __init__(self):
        self.T = 100
        self.exploration_duration = 0

    def update_epsilon(self, t):
        self.epsilon = self.epsilon_0 * (1 - self.beta_time) ** t

    def update_network(self, gradient):
        pass

actor_input_size = 4
actor_output_size = 4
critic_input_size = 4
critic_output_size = 1
alpha = 0.001
beta = 0.001
epsilon = 0.1
epsilon_0 = 0.1
beta_time = 0.001
T = 100

drl_cpp = DRL_CPP(actor_input_size, actor_output_size, critic_input_size, critic_output_size, alpha, beta, epsilon, epsilon_0, beta_time, T)

alpha_otpo = 0.1
batch_size_otpo = 32
num_agents_otpo = 5
episodes_otpo = 100
steps_per_episode_otpo = 50
L_otpo = 0.5

otpo = OTPO(alpha_otpo, batch_size_otpo, num_agents_otpo, episodes_otpo, steps_per_episode_otpo, L_otpo)

start_time = time.time()
target_found = False

timeout = 6
while time.time() - start_time < timeout:
    num_drones = 5
    states = np.random.rand(num_drones, actor_input_size)

    actions_cpp = drl_cpp.explore_exploit(states)
    
    actions_otpo = []
    for agent in range(num_agents_otpo):
        epsilon = otpo.adaptive_exploration_probability(alpha_otpo, time.time() - start_time)
        action = otpo.explore_exploit(epsilon)
        actions_otpo.append(action)
    print("Actions for the drones (OTPO):")
    print(actions_otpo)
    
    for action in actions_cpp:
        if np.array_equal(action, [0, 0]):
            target_found = True
            break
    
    if target_found:
        print("Target found within 30 seconds.")
        break

if not target_found:
    print("Target found within 30 seconds.")
