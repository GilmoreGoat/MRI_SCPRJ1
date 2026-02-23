
import importlib

packages = ['numpy', 'torch', 'torch_geometric', 'networkx', 'nilearn']
missing = []

for package in packages:
    try:
        importlib.import_module(package)
        print(f"{package} is installed.")
    except ImportError:
        print(f"{package} is NOT installed.")
        missing.append(package)

if missing:
    print(f"Missing packages: {missing}")
else:
    print("All packages are installed.")
