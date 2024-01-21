# Crop-and-Weed-Segmentation-UsingDeepLabV3Plus-Method

# Steps to get the project running

1. Clone repository
2. Install requirements.txt

# How to infere with segmentation model

## Installation
An environment with a pre-configured Python + Torch installation using GPUs is available. Please follow
[this link.](https://dept-info.labri.fr/~mansenca/public/CREMI_deeplearning_env.pdf)

Then, you have to add the [src](src) folder to your python path, so it can find the given API. In Linux
it is done by running the following command, after replacing with the correct path (e.g., using pwd command
in src folder): 
```bash
export PYTHONPATH = $PYTHONPATH:/absolute/path/to/src
```
Note that this should be done everytime if you are running scripts in a terminal.

In a Jupyter Notebook (and even in a script), it can be done manually as follows:
```python
import sys
sys.path.append("/absolute/path/to/src")
```
before doing any import of the am4ip library.

One can also configure its IDE for the project.

### In Pycharm:
- right Click on the src folder
- Mark Directory as > Sources Root
