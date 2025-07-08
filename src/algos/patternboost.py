"""
PatternBoost Algorithm Implementation
Implements the complete PatternBoost process for grid configuration optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import os
import pickle

from src.models.transformer_model import PatternBoostTransformer, GridTokenizer, create_transformer_config
from src.envs.base_env import GridSubsetEnvWithPriority
from src.registry.algo_registry import get_algo


class PatternBoost:
    """
    PatternBoost algorithm for grid configuration optimization
    
    The complete process follows these steps:
    1. Create initial dataset using greedy search
    2. Train transformer on top configurations
    3. Generate new seeds using transformer
    4. Run local search on new seeds
    5. Iterate and improve
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (5, 5),
                 device: str = 'cpu',
                 model_dir: str = './saved_models',
                 env=None,
                 **kwargs):
        self.grid_size = grid_size
        self.device = device
        self.model_dir = model_dir
        self.m, self.n = grid_size
        
        # Initialize tokenizer
        self.tokenizer = GridTokenizer(grid_size)
        
        # Initialize transformer
        config = create_transformer_config(grid_size, **kwargs)
        self.transformer = PatternBoostTransformer(config).to(device)
        
        # Use provided environment or create default one
        if env is not None:
            self.env = env
        else:
            self.env = GridSubsetEnvWithPriority(m=self.m, n=self.n)
        
        # Local search method mapping
        self.local_search_methods = {
            'greedy': self._greedy_search,
            'mcts_basic': self._mcts_search,
            'mcts_priority': self._mcts_search,
            'mcts_parallel': self._mcts_search,
            'mcts_advanced': self._mcts_search,
        }
        
        # Get local search method from config
        self.local_search_method = kwargs.get('local_search_method', 'greedy')
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.betas = kwargs.get('betas', (0.9, 0.95))
        
        # PatternBoost parameters
        self.initial_dataset_size = kwargs.get('initial_dataset_size', 10000)
        self.top_percentage = kwargs.get('top_percentage', 0.25)
        self.generation_size = kwargs.get('generation_size', 1000)
        self.max_iterations = kwargs.get('max_iterations', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs_per_iteration = kwargs.get('epochs_per_iteration', 10)
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def _greedy_search(self, config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Run greedy search to generate a configuration"""
        self.env.reset()
        
        # Run greedy search using environment method
        num_points = self.env.greedy_search(random=True)
        
        # Get final state and score
        final_state = self.env.state.copy()
        score = self._calculate_score(final_state)
        
        return final_state, score
    
    def _mcts_search(self, config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Run MCTS search to generate a configuration"""
        if config is None:
            config = {
                'num_searches': 1000,
                'C': 1.4,
                'top_n': 1
            }
        
        # Run MCTS using environment method
        self.env.reset()
        num_points = self.env.mcts_basic(config)
        
        # Get final state and score
        final_state = self.env.state.copy()
        score = self._calculate_score(final_state)
        
        return final_state, score
    
    def _calculate_score(self, grid: np.ndarray) -> float:
        """Calculate score for a grid configuration"""
        # Count number of points placed
        num_points = np.sum(grid)
        
        # For now, use simple scoring based on number of points
        # In practice, you might want to check for collinearity violations
        return float(num_points)
    
    def _create_initial_dataset(self) -> List[Tuple[np.ndarray, float]]:
        """Step 1: Create initial dataset using greedy search"""
        print(f"Creating initial dataset with {self.initial_dataset_size} configurations...")
        
        dataset = []
        for i in tqdm(range(self.initial_dataset_size), desc="Generating initial dataset"):
            # Use greedy search for initial dataset
            grid, score = self._greedy_search()
            dataset.append((grid, score))
        
        # Sort by score (descending)
        dataset.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Initial dataset created. Best score: {dataset[0][1]}")
        return dataset
    
    def _train_transformer(self, training_grids: List[np.ndarray], 
                          epochs: int = 10) -> float:
        """Step 2: Train transformer on top configurations"""
        print(f"Training transformer on {len(training_grids)} configurations...")
        
        # Create training data
        x, y = self.tokenizer.create_training_data(training_grids)
        x, y = x.to(self.device), y.to(self.device)
        
        # Setup optimizer
        optimizer = self.transformer.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=self.betas,
            device_type=self.device
        )
        
        # Training loop
        self.transformer.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(x), self.batch_size):
                batch_x = x[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                
                # Forward pass
                logits, loss = self.transformer(batch_x, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            total_loss += avg_loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        avg_total_loss = total_loss / epochs
        print(f"Training completed. Average loss: {avg_total_loss:.4f}")
        return avg_total_loss
    
    def _generate_new_seeds(self, num_configs: int = 1000) -> List[np.ndarray]:
        """Step 3: Generate new seeds using trained transformer"""
        print(f"Generating {num_configs} new configurations...")
        
        self.transformer.eval()
        new_configs = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_configs), desc="Generating configurations"):
                # Start with start token
                start_token = torch.tensor([[self.tokenizer.start_token]], 
                                         dtype=torch.long, device=self.device)
                
                # Generate sequence
                generated = self.transformer.generate(
                    start_token,
                    max_new_tokens=50,  # Maximum tokens to generate
                    temperature=1.0,
                    do_sample=True,
                    top_k=10
                )
                
                # Convert to grid
                tokens = generated[0].cpu().numpy().tolist()
                grid = self.tokenizer.decode_tokens(tokens)
                new_configs.append(grid)
        
        print(f"Generated {len(new_configs)} new configurations")
        return new_configs
    
    def _run_local_search(self, configs: List[np.ndarray], 
                         method: Optional[str] = None) -> List[Tuple[np.ndarray, float]]:
        """Step 4: Run local search on generated configurations"""
        if method is None:
            method = self.local_search_method
            
        print(f"Running {method} local search on {len(configs)} configurations...")
        
        improved_configs = []
        
        for config in tqdm(configs, desc=f"Running {method} search"):
            # Set environment to initial state
            self.env.reset()
            
            # Apply the configuration to environment
            for i in range(self.m):
                for j in range(self.n):
                    if config[i, j] == 1:
                        point = self.env.decode_action(i * self.n + j)
                        self.env.add_point(point)
            
            # Run local search using the selected method
            if method in self.local_search_methods:
                improved_grid, score = self.local_search_methods[method]()
            else:
                # Default to greedy if method not found
                improved_grid, score = self._greedy_search()
            
            improved_configs.append((improved_grid, score))
        
        print(f"Local search completed. Best improved score: {max(c[1] for c in improved_configs)}")
        return improved_configs
    
    def _save_model(self, iteration: int):
        """Save transformer model"""
        model_path = os.path.join(self.model_dir, f'patternboost_transformer_iter_{iteration}.pt')
        
        # Save config as a dictionary instead of the object
        config_dict = {
            'vocab_size': self.transformer.config.vocab_size,
            'block_size': self.transformer.config.block_size,
            'n_layer': self.transformer.config.n_layer,
            'n_head': self.transformer.config.n_head,
            'n_embd': self.transformer.config.n_embd,
            'dropout': self.transformer.config.dropout,
            'bias': self.transformer.config.bias
        }
        
        torch.save({
            'model_state_dict': self.transformer.state_dict(),
            'config_dict': config_dict,
            'iteration': iteration
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def _load_model(self, iteration: int):
        """Load transformer model"""
        model_path = os.path.join(self.model_dir, f'patternboost_transformer_iter_{iteration}.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            return True
        return False
    
    def _save_dataset(self, dataset: List[Tuple[np.ndarray, float]], 
                     iteration: int):
        """Save dataset"""
        dataset_path = os.path.join(self.model_dir, f'patternboost_dataset_iter_{iteration}.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {dataset_path}")
    
    def _load_dataset(self, iteration: int) -> Optional[List[Tuple[np.ndarray, float]]]:
        """Load dataset"""
        dataset_path = os.path.join(self.model_dir, f'patternboost_dataset_iter_{iteration}.pkl')
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Dataset loaded from {dataset_path}")
            return dataset
        return None
    
    def run(self, resume_from: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete PatternBoost process
        
        Args:
            resume_from: If provided, resume from this iteration
            
        Returns:
            Dictionary with results and statistics
        """
        print("Starting PatternBoost algorithm...")
        
        # Initialize dataset
        if resume_from is not None and resume_from > 0:
            dataset = self._load_dataset(resume_from - 1)
            if dataset is None:
                print(f"Could not load dataset for iteration {resume_from - 1}, starting from scratch")
                dataset = self._create_initial_dataset()
        else:
            dataset = self._create_initial_dataset()
        
        # Load model if resuming
        if resume_from is not None and resume_from > 0:
            if not self._load_model(resume_from - 1):
                print(f"Could not load model for iteration {resume_from - 1}, starting from scratch")
        
        results = {
            'iterations': [],
            'best_scores': [],
            'dataset_sizes': [],
            'training_losses': []
        }
        
        start_iteration = resume_from if resume_from is not None else 0
        
        # Track best-ever configuration and score
        best_ever_score = -float('inf')
        best_ever_config = None
        
        for iteration in range(start_iteration, self.max_iterations):
            print(f"\n=== PatternBoost Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # Select top configurations for training
            num_top = max(1, int(len(dataset) * self.top_percentage))
            top_configs = [config for config, _ in dataset[:num_top]]
            
            print(f"Selected top {len(top_configs)} configurations for training")
            
            # Train transformer
            training_loss = self._train_transformer(top_configs, self.epochs_per_iteration)
            
            # Generate new seeds
            new_configs = self._generate_new_seeds(self.generation_size)
            
            # Run local search on new seeds
            improved_configs = self._run_local_search(new_configs, method='greedy')
            
            # Track best-ever configuration and score
            for config, score in improved_configs:
                if score > best_ever_score:
                    best_ever_score = score
                    best_ever_config = config
            
            # Add improved configurations to dataset
            dataset.extend(improved_configs)
            
            # Sort by score and keep top configurations
            dataset.sort(key=lambda x: x[1], reverse=True)
            dataset = dataset[:self.initial_dataset_size]  # Keep dataset size manageable
            
            # Save model and dataset
            self._save_model(iteration)
            self._save_dataset(dataset, iteration)
            
            # Record results
            best_score = dataset[0][1] if dataset else 0
            results['iterations'].append(iteration)
            results['best_scores'].append(best_score)
            results['dataset_sizes'].append(len(dataset))
            results['training_losses'].append(training_loss)
            
            print(f"Iteration {iteration + 1} completed:")
            print(f"  Best score: {best_score}")
            print(f"  Dataset size: {len(dataset)}")
            print(f"  Training loss: {training_loss:.4f}")
        
        print("\n=== PatternBoost completed ===")
        print(f"Final best score: {results['best_scores'][-1]}")
        print(f"Best-ever score: {best_ever_score}")
        
        # Save best-ever configuration and score for later use
        self.best_ever_score = best_ever_score
        self.best_ever_config = best_ever_config
        
        return results
    
    def generate_best_configuration(self) -> Tuple[np.ndarray, float]:
        """Generate the best configuration using the trained model"""
        print("Generating best configuration...")
        
        # If best-ever config is available, return it
        if hasattr(self, 'best_ever_config') and self.best_ever_config is not None:
            print(f"Returning best-ever configuration with score: {self.best_ever_score}")
            return self.best_ever_config, self.best_ever_score
        
        # Otherwise, fallback to generating new configs
        configs = self._generate_new_seeds(100)
        improved_configs = self._run_local_search(configs, method='greedy')
        best_config, best_score = max(improved_configs, key=lambda x: x[1])
        print(f"Best configuration found with score: {best_score}")
        return best_config, best_score 