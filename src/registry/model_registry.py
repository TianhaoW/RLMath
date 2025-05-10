from src.models import FFQNet, ConvQNet

MODEL_CLASSES = {
    "ffnn": lambda grid_shape, output_dim: FFQNet(grid_shape, output_dim),
    "cnn": lambda grid_shape, output_dim: ConvQNet(grid_shape, output_dim),
}