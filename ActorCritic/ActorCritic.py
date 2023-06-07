
from tqdm import tqdm

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

# For testing and running in main
import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, continuous_action_space=False, action_std_dev=None, lr=0.001):
        super(Actor, self).__init__()
        # Setting up for continuous spaces
        self.continuous_action_space = continuous_action_space
        if continuous_action_space:
            # Creates a variance tensor given a starting standard deviation of actions
            # This is required for continuous action spaces
            self.action_var = torch.full((action_space_dim,), action_std_dev * action_std_dev).to(DEVICE)
            self.action_space_dim = action_space_dim
            # Setup actor
            # Final output should be an action (that's why this is an ACTor), actor network is a policy for action given state
            self.actor = nn.Sequential(nn.Linear(state_space_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_space_dim),
                            nn.Tanh())
        else:
            # If not continous setup actor slightly different
            self.actor = nn.Sequential(nn.Linear(state_space_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_space_dim),
                            nn.Softmax(dim=-1)) # We want it to be one be discrete actions spaces are one dimensional
        
        self.optim = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
        return self.actor(x) # Returns "probs" if discrete and mean if continuous
    
    def update_action_var(self, new_action_std_dev):
        """
        Updates the action variance based on a newly measured action standard deviation.
        Required for using continuous actions.
        """
        assert self.continuous_action_space == True, "Attempting to update action variance in discrete action space."
        self.action_var = torch.full((self.action_space_dim,), new_action_std_dev * new_action_std_dev).to(DEVICE)

    def act(self, state):
        """
        Given state returns action and action log probability
        
        Input:
            state
        Output:
            action: discrete or continuous
            action_logprob: used for calculating errors and updating
        """
        # Get an action distribution (dependent on action space being continous or discrete)
        if self.continuous_action_space:
            action_mean = self.forward(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.forward(state)
            dist = Categorical(action_probs)  

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def loss(self, state_value, new_state_value, reward, log_prob, discount, I):
        advantage = reward + discount * new_state_value.item() - state_value.item()
        policy_loss = -log_prob * advantage
        policy_loss *= I
        return policy_loss
    
    def backprop(self, policy_loss):
        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()
                  
class Critic(nn.Module):
    def __init__(self, state_space_dim, lr=0.001):
        super(Critic, self).__init__()
        # Critic is a state value estimator, it doesn't really care about actions
        self.critic = nn.Sequential(nn.Linear(state_space_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1))
        
        self.optim = optim.SGD(self.parameters(), lr=lr)
        
    def forward(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
        state_value = self.critic(x)
        return state_value

    def loss(self, state_value, new_state_value, reward, discount, I):
        val_loss = F.mse_loss(reward + discount * new_state_value, state_value)
        val_loss *= I
        return val_loss

    def backprop(self, state_value_loss):
        self.optim.zero_grad()
        state_value_loss.backward()
        self.optim.step()

class ActorCritic():
    def __init__(self, state_space_dim, action_space_dim, continuous_action_space=False, action_std_dev=None, actor_lr=0.001, critic_lr=0.001, num_episodes=1000, max_steps=10000, discount=0.99):
        self.actor = Actor(state_space_dim, action_space_dim, continuous_action_space, action_std_dev, actor_lr)
        self.critic = Critic(state_space_dim, critic_lr)

        # Network parameters
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.discount = discount

    def get_action(self, state):
        '''
        Given state return action, action log probability, and state's value.
        '''
        action, action_logprob = self.actor.act(state)
        state_val = self.critic.forward(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def loss(self, state_value, new_state_value, reward, action_logprob, I):
        state_value_loss = self.critic.loss(state_value, new_state_value, reward, self.discount, I)
        policy_loss = self.actor.loss(state_value, new_state_value, reward, action_logprob, self.discount, I)
        return state_value_loss, policy_loss
    
    def backprop(self, state_value_loss, policy_loss):
        self.actor.backprop(policy_loss)
        self.critic.backprop(state_value_loss)

    def train(self, env):
        scores = []
        for episode in tqdm(range(self.num_episodes)):
            # Reset everything to starting states
            state = env.reset()
            # This needs to be done for the gym state 
            state = state[0]
            done = False
            score = 0
            I = 1
            for step in range(self.max_steps):
                action, action_logprob, state_value = self.get_action(state)
                new_state, reward, done, _ = env.step(action)
                score += reward
                # If the action finished the task, the next state value is zero! Don't go further you're done!
                if done:
                    new_state_value = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)
                else:
                    new_state_value = self.critic.forward(new_state)

                state_value_loss, policy_loss = self.loss(state_value, new_state_value, reward, action_logprob, I)
                #Backpropagate policy
                self.backprop(state_value_loss, policy_loss)
        
                if done:
                    break
            
                # Transition to new state
                state = new_state
                I *= self.discount
            scores.append(score)
        return scores

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    actorcritic = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    scores = actorcritic.train(env)

    sns.set()

    plt.plot(scores)
    plt.ylabel('score')
    plt.xlabel('episodes')
    plt.title('Training score of CartPole Actor-Critic TD(0)')

    reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
    y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
    plt.plot(y_pred)
    plt.show()



