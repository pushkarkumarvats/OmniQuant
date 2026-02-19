"""RL agents for adaptive market making and execution."""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from loguru import logger


class TradingEnvironment(gym.Env):
    """Gym environment for market-making RL.

    State: [inventory, volatility, spread, OFI, time_remaining, position_pnl]
    Action: [bid_offset, ask_offset, aggressiveness]
    Reward: PnL - lambda * risk²
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # State space: [inventory, volatility, spread, OFI, time, pnl]
        self.observation_space = spaces.Box(
            low=np.array([-1000, 0, 0, -1, 0, -10000]),
            high=np.array([1000, 1, 1, 1, 1, 10000]),
            dtype=np.float32
        )
        
        # Action space: [bid_offset, ask_offset, aggressiveness]
        self.action_space = spaces.Box(
            low=np.array([-0.01, -0.01, 0]),
            high=np.array([0.01, 0.01, 1]),
            dtype=np.float32
        )
        
        # Environment state
        self.inventory = 0
        self.cash = 0
        self.position_pnl = 0
        self.mid_price = 100.0
        self.spread = 0.01
        self.volatility = 0.01
        self.ofi = 0.0
        self.time_step = 0
        self.max_steps = 390  # Trading day
        
        # Risk parameters
        self.risk_aversion = config.get('risk_aversion', 0.01)
        self.inventory_penalty = config.get('inventory_penalty', 0.001)
    
    def reset(self) -> np.ndarray:
        self.inventory = 0
        self.cash = 0
        self.position_pnl = 0
        self.mid_price = 100.0
        self.time_step = 0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.inventory / 1000,  # Normalized
            self.volatility,
            self.spread,
            self.ofi,
            self.time_step / self.max_steps,
            self.position_pnl / 1000
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Simulate one tick: execute orders, update market state, compute reward."""
        bid_offset, ask_offset, aggressiveness = action
        
        # Simulate market dynamics
        price_change = np.random.randn() * self.volatility
        self.mid_price += price_change
        
        # Update market microstructure
        self.spread = max(0.001, self.spread + np.random.randn() * 0.0001)
        self.ofi = np.clip(np.random.randn() * 0.1, -1, 1)
        
        # Simulate order execution
        bid_price = self.mid_price - self.spread/2 + bid_offset
        ask_price = self.mid_price + self.spread/2 + ask_offset
        
        # Probability of fill based on aggressiveness
        bid_fill_prob = min(0.9, aggressiveness)
        ask_fill_prob = min(0.9, aggressiveness)
        
        # Execute trades
        if np.random.rand() < bid_fill_prob:
            # Buy at bid
            trade_size = np.random.randint(1, 100)
            self.inventory += trade_size
            self.cash -= bid_price * trade_size
        
        if np.random.rand() < ask_fill_prob:
            # Sell at ask
            trade_size = np.random.randint(1, 100)
            self.inventory -= trade_size
            self.cash += ask_price * trade_size
        
        # Calculate reward
        self.position_pnl = self.cash + self.inventory * self.mid_price
        
        reward = (
            self.position_pnl / 1000  # PnL component
            - self.risk_aversion * (self.inventory ** 2)  # Risk penalty
            - self.inventory_penalty * abs(self.inventory)  # Inventory penalty
        )
        
        # Update time
        self.time_step += 1
        done = self.time_step >= self.max_steps
        
        info = {
            'inventory': self.inventory,
            'pnl': self.position_pnl,
            'mid_price': self.mid_price,
            'spread': self.spread
        }
        
        return self._get_observation(), reward, done, info


class DQNAgent:
    """DQN with experience replay and a lagged target network."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        
        # Networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=config.get('memory_size', 10000))
    
    def _build_network(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Epsilon-greedy action selection (greedy when not training)."""
        if training and np.random.rand() < self.epsilon:
            # Random action
            return np.random.uniform(-0.01, 0.01, self.action_dim)
        
        # Greedy action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Convert Q-values to continuous action
        action = torch.tanh(q_values).cpu().numpy()[0] * 0.01
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q.max(1)[0]
        
        # Loss
        loss = nn.MSELoss()(current_q.max(1)[0], target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        logger.info(f"Model loaded from {path}")


class RLMarketMaker:
    """Wraps DQNAgent + TradingEnvironment for market-making training loops."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.env = TradingEnvironment(config)
        self.agent = DQNAgent(
            state_dim=6,
            action_dim=3,
            config=config
        )
        self.episode_rewards: List[float] = []
    
    def train(self, num_episodes: int = 1000):
        """Run the training loop for the given number of episodes."""
        logger.info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select and execute action
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train
                self.agent.replay()
                
                state = next_state
                episode_reward += reward
            
            # Update target network
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            self.episode_rewards.append(episode_reward)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )
        
        logger.info("Training complete!")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Roll out the learned policy and return average PnL and trade count."""
        total_pnl = 0
        total_trades = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_pnl = 0
            
            while not done:
                action = self.agent.act(state, training=False)
                state, reward, done, info = self.env.step(action)
                episode_pnl = info['pnl']
            
            total_pnl += episode_pnl
            total_trades += self.env.time_step
        
        return {
            'avg_pnl': total_pnl / num_episodes,
            'avg_trades': total_trades / num_episodes
        }
    
    def get_action(self, state: np.ndarray) -> Dict[str, float]:
        """Return bid/ask offsets and aggressiveness for the given state."""
        action = self.agent.act(state, training=False)
        return {
            'bid_offset': action[0],
            'ask_offset': action[1],
            'aggressiveness': action[2]
        }


class RLExecutionAgent:
    """DQN agent that learns optimal order slicing and timing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Execution-specific environment
        self.env = TradingEnvironment(config)
        self.agent = DQNAgent(
            state_dim=6,
            action_dim=2,  # [slice_size, aggressiveness]
            config=config
        )
    
    def train(self, num_episodes: int = 1000):
        """Train execution agent"""
        logger.info(f"Training RL Execution Agent for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay()
                
                state = next_state
            
            if episode % 10 == 0:
                self.agent.update_target_network()
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Epsilon={self.agent.epsilon:.3f}")
    
    def get_optimal_execution(
        self,
        total_quantity: float,
        time_horizon: int,
        market_state: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Produce a schedule of child-order slices to fill *total_quantity*."""
        remaining = total_quantity
        execution_schedule = []
        
        for t in range(time_horizon):
            action = self.agent.act(market_state, training=False)
            slice_size = min(remaining, abs(action[0]) * total_quantity)
            
            execution_schedule.append({
                'time': t,
                'quantity': slice_size,
                'aggressiveness': action[1]
            })
            
            remaining -= slice_size
            if remaining <= 0:
                break
        
        return execution_schedule


if __name__ == "__main__":
    # Test RL agents
    logger.info("Testing RL Market Maker...")
    
    config = {
        'risk_aversion': 0.01,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 0.001,
        'batch_size': 64
    }
    
    # Train market maker
    mm = RLMarketMaker(config)
    mm.train(num_episodes=500)
    
    # Evaluate
    metrics = mm.evaluate(num_episodes=10)
    logger.info(f"Evaluation Results: {metrics}")
    
    # Test execution agent
    logger.info("\nTesting RL Execution Agent...")
    exec_agent = RLExecutionAgent(config)
    exec_agent.train(num_episodes=500)
    
    # Get optimal execution
    state = np.array([0, 0.01, 0.01, 0, 0, 0])
    schedule = exec_agent.get_optimal_execution(
        total_quantity=10000,
        time_horizon=10,
        market_state=state
    )
    logger.info(f"Optimal execution schedule: {schedule}")
    
    logger.info("\nRL agents tested successfully!")
