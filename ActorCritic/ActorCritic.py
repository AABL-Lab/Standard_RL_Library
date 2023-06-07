import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, continuous_action_space=False, action_std_dev=None):
        super(Actor, self).__init__()
        # Initial state and action parameters
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim

        # Setting up for continuous spaces
        self.continuous_action_space = continuous_action_space
        if continuous_action_space:
            # Creates a variance tensor given a starting standard deviation of actions
            # This is required for continuous action spaces
            self.action_var = torch.full((action_space_dim,), action_std_dev * action_std_dev).to(DEVICE)
            
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
    
    def forward(self, x):
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
                  
class Critic(nn.Module):
    def __init__(self, state_space_dim, ):
        super(Actor, self).__init__()
        # Initial state and action parameters
        self.state_space_dim = state_space_dim

        # Critic is a state value estimator, it doesn't really care about actions
        self.critic = nn.Sequential(nn.Linear(state_space_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1))
        
    def forward(self, x):
        state_values = self.critic(x)
        return state_values

class ActorCritic():
    def __init__(self, state_space_dim, action_space_dim, continuous_action_space=False, action_std_dev=None):
        self.actor = Actor(state_space_dim, action_space_dim, continuous_action_space, action_std_dev)
        self.critic = Critic(state_space_dim)

    def get_action(self, state):
        '''
        Given state return action, action log probability, and state's value.
        '''
        action, action_logprob = self.actor.act(state)
        state_val = self.critic.forward(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

