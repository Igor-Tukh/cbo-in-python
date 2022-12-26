# Demos

To highlight the possibilities of the library, and provide some additional examples, we prepared a few demo scripts.

## CBO in PyTorch

### Neural Networks training

`torch_nn_demo.py` provides a command line interface for training a few standard neural networks on the `MNIST` dataset and visualizing the training progress.

#### Instructions

* Follow the [setup instructions](https://github.com/Igor-Tukh/cbo-in-python/blob/master/README.md) to prepare the environment.
* Run the CLI with standard command line arguments:
```
python3 torch_nn_demo.py
```
* Use `--help` flag to view the information about command-line arguments:
```
python3 torch_nn_demo.py --help
```

CLI provides many arguments to configure the training. Below we provide a few main arguments:

| **argument** | **description** | **type** |
| --- | --- | --- |
| --model | architecture to use | str |
| --dataset | dataset to use | str |
| --device | cpu / cuda | str |
| --epochs | # training epochs | int |
| --batch_size | batch size (for samples-level batching) | int |
| --build_plot | specify to build loss and accuracy plot | - |

CLI provides many more arguments. Please refer to `--help` instructions for more detailed information:
```
python3 torch_nn_demo.py --help
```

For instance, to train LeNet5 for ten epochs and save the resulting plots, one may use the following command:
```
python3 torch_nn_demo.py --epochs 10 --build_plot --model LeNet5 --plot_path result.png
```
