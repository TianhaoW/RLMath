from src.models.dqn_models import FFQNet, ConvQNet, ViTQNet

MODEL_CLASSES = {
    "ffnn": lambda grid_shape, output_dim: FFQNet(grid_shape, output_dim),
    "cnn": lambda grid_shape, output_dim: ConvQNet(grid_shape, output_dim),
    "vit": lambda grid_shape, output_dim: ViTQNet(grid_shape, output_dim),
}