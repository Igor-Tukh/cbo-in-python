class Loss:
    def __init__(self, loss, optimizer):
        """
        Wrapper for the ordinary loss to use in the CBO optimization. Mainly purposed for making the CBO optimization
        training loop close to the typical `PyTorch` training loop.
        :param loss: ordinary loss function.
        :param optimizer: consensus-based optimizer.
        """
        optimizer.set_loss(loss)
        self.loss = loss
        self.optimizer = optimizer

    def __call__(self, outputs, targets):
        """
        Returns the values of the underlying loss function computed on the passed arguments.
        :param outputs: model outputs.
        :param targets: target values.
        :return: the values of the underlying loss function.
        """
        return self.loss(outputs, targets)

    def backward(self, X, y, backward_gradients=False):
        """
        Updates the CB optimizer state. If `backward_gradients` is specified, applies the regular backpropagation
        algorithm (based on the underlying loss function values).
        :param X: updates the data batch values to use in the optimization.
        :param y: updates the data batch targets (labels) to use in the optimization.
        :param backward_gradients:
        """
        self.optimizer.set_batch(X, y)
        if backward_gradients:
            self.optimizer.backward(self.loss)

    def set_data_batch(self, X, y):
        """
        Alias for the `backward` function. Doesn't call the backpropagation algorithm.
        """
        self.backward(X, y)
