## User Guide

## Contribution Guide

### Environment Setup (using Poetry)

1. **Install [Poetry](https://python-poetry.org/docs/#installation):**

   ```bash
   pip install poetry
   
   # Check if sucessfully installed
   poetry -V
   ```

2. **Install project dependencies:**

   ```bash
   poetry install
   ```

3. **Testing your environment:**

   Use the following command to see if you set up your environment correctly or not

   ```bash
   poetry run python tests/env_check.py
   ```

---

### 🧠 Implementing and Registering New Models

1. **Implement your model class** in [`src/models.py`](src/models/models.py). Your model should:

   - Inherit from `nn.Module`.
   - Accept `(grid_shape, output_dim)` in the constructor.
   - Accept raw `(B, m, n)` input and handle encoding internally in the `forward()` method.

2. **Register your model** in [`src/registry/model_registry.py`](src/registry/model_registry.py):

   ```python
   from src.models.models import MyNewModel

   MODEL_CLASSES = {
       "cnn": lambda grid_shape, output_dim: ConvQNet(grid_shape, output_dim),
       "flat": lambda grid_shape, output_dim: FFQNet(grid_shape, output_dim),
       "my_model": lambda grid_shape, output_dim: MyNewModel(grid_shape, output_dim),
   }
   ```

3. **Update the configuration** in `config.toml`:

   ```toml
   [env]
   model = "my_model"
   ```

---

### Implementing and Registering New Environments

1. **Implement your environment class** as a subclass of `GridSubsetEnv` in a new file within the `src/` directory. Override the `add_point(self, point: Point)` method to define your constraint logic.

2. **Register your environment** in [`src/registry/env_registry.py`](src/registry/env_registry.py):

   ```python
   from src.my_env_file import MyNewEnv

   ENV_CLASSES = {
       "NoIsoscelesEnv": NoIsoscelesEnv,
       "NoStrictIsoscelesEnv": NoStrictIsoscelesEnv,
       "MyNewEnv": MyNewEnv,
   }
   ```

3. **Update the configuration** in `config.toml`:

   ```toml
   [env]
   env_type = "MyNewEnv"
   ```

---

###  Configuration Management via `config.toml`

All project configurations are centralized in the `config.toml` file, including:

- **Grid size**: `m` and `n`.
- **Environment type**: `env_type`.
- **Model type**: `model`.
- **Training hyperparameters**: learning rate, epsilon, gamma, etc.
- **Paths**: directories for logs and saved models.

**Example `config.toml` snippet:**

```toml
# parameters for the enviroment and model
[env]
m = 4                              # this is the number of rows of the grid
n = 4                              # this is the number of columns of the grid
model = "ffnn"                     # the available models are in src/registry/model_registry.py
env_type = "NoStrictIsoscelesEnv"  # the available envs are in src/registry/env_registry.py

# parameters for training
[train]
episodes = 5000                    # the number of training epsisodes
batch_size = 64
gamma = 1.0                        # Gamma is the discount factor for reward. We set it to 1 for no discount.
epsilon = 0.05                     # epsilon is the probability for exploring
lr = 0.0001                        # This is the learning rate
target_update_freq = 10
memory_size = 10000
save_best_points = true  # save and print best point set at the end


# all the path are relative to the project root
[path]
project_root = './'
log_dir = './logs/'
model_dir = './models/'
```

---

### Utilizing Configuration and Logger in Custom Scripts

To access the configuration and logger in your scripts:

```python
from src.utils import parse_config, get_logger

# Parse the configuration
config = parse_config()

# Initialize the logger
logger = get_logger("your_script_name", config)
```

- **`parse_config()`**: Parses the `config.toml` file and returns a configuration dictionary.
- **`get_logger(name, config)`**: Initializes and returns a logger with the specified name. Logs are saved in a timestamped directory within the `logs/` folder.

**Example usage:**

```python
logger.info("This is an informational message.")
logger.error("This is an error message.")
```

---

### 📁 Output Directories

- **Models**: Saved in the `models/` directory.
  - Filename format: `<EnvType>_<ModelType>_<GridSize>.pt`
  - Example: `NoStrictIsoscelesEnv_cnn_5x10.pt`

- **Logs**: Saved in the `logs/` directory.
  - Each run creates a timestamped subdirectory: `logs/run_<timestamp>/`
  - Contains log files for training and testing sessions.

---

### 🧪 Running Training and Evaluation

- **Train a model**:

  ```bash
  poetry run python train_model.py
  ```

- **Evaluate a saved model**:

  ```bash
  poetry run python test_model.py
  ```

Ensure that the desired configurations are set in the `config.toml` file before running these scripts.

---



## File Structure
```
RLMath/
├── config.toml               # All training testing & environment configs (hyperparams, paths, etc.)
├── pyproject.toml            # Poetry-managed project metadata and dependencies
├── poetry.lock               # Exact dependency versions for reproducibility
├── README.md                 # Project overview and usage instructions

├── train_model.py            # Main training script
├── test_model.py             # Evaluation-only script (greedy policy)

├── logs/                     # Auto-generated logs per run (e.g., logs/run_2025-05-09_12-00/)
│   └── run_<timestamp>/      # Contains log files per training run

├── saved_models/                   # Saved model checkpoints
│   └── <env>_<model>_<grid>.pt  # e.g., NoStrictIsoscelesEnv_ffnn_5x10.pt

├── src/                      # Source code package
│   ├── __init__.py           # Set up file for the code package    
│   ├── utils.py              # Config parser, logging setup, helpers
│   ├── env/  
│       ├── base_env.py           # Gym-style abstract GridSubsetEnv base class
│       ├── isosceles_triangle.py # Implements NoIsoscelesEnv, NoStrictIsoscelesEnv class
│   ├── models/  
│       ├── models.py             # E.g FFQNet (flat) and ConvQNet (cnn) model definitions
│   └── registry/             # Registry pattern for envs and models
│       ├── env_registry.py   # Maps env names to classes
│       └── model_registry.py # Maps encoding to model classes

└── tests/                    # Saved notedbooks and .py files for testing the environment 
```