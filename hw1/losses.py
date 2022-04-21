import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        num_train = x.shape[0]
        margin = x_scores - x_scores[range(num_train), y].reshape(-1, 1) + 1
        margin[range(num_train), y] = 0
        margin1 = torch.sum(margin, 1)
        # margin-=1

        loss = torch.sum(margin1[margin1 > 0])
        loss /= num_train
        # ========================



        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["num_train"] = num_train
        self.grad_ctx["margin"] = margin
        self.grad_ctx["x"] = x
        self.grad_ctx["y"] = y
        # ========================

        return loss.reshape([1])

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.
        # Wk = Wk-1 - nk * gradW * L(Wk-1)
        # gradW * L(Wk-1) size = D+1 * C = features * n_classes
        grad = None
        # ====== YOUR CODE: ======
        num_train = self.grad_ctx["num_train"]
        margin = self.grad_ctx["margin"]
        x = self.grad_ctx["x"]
        y = self.grad_ctx["y"]
        margin2 = margin.clone()
        margin2[margin2 > 0] = 1
        margin2[margin2 <= 0] = 0
        margin2[range(num_train), y] = -1
        margin3 = margin.clone()
        margin3[margin2 > 0] = 1
        margin3[margin2 <= 0] = 0
        margin3[range(num_train), y] = torch.sum(margin3, axis=1) * -1
        dW = x.t() @ margin3
        dW /= num_train
        grad = dW

        # ========================

        return grad
