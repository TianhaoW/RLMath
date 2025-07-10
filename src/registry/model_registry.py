from src.models.dqn_models import FFQNet, ConvQNet, ViTQNet

MODEL_CLASSES = {
    "ffnn": lambda grid_shape, output_dim, channels: FFQNet(grid_shape, output_dim, channels),
    "cnn": lambda grid_shape, output_dim, channels: ConvQNet(grid_shape, output_dim, channels),
    "vit": lambda grid_shape, output_dim, channels: ViTQNet(grid_shape, output_dim, channels),
}