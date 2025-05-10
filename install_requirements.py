import toml
from pathlib import Path

pyproject = Path(__file__).resolve().parent / "pyproject.toml"
config = toml.load(pyproject)

packages = config['tool']['poetry']['dependencies']
del packages['python']  # skip version pin

print("To install dependencies, run:")
for pkg in packages:
    print(f"pip install {pkg}")