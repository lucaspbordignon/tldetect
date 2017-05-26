import tensorflow as tf


class Convolutional_layer():
    """
        A layer that computes the convolutions on a CNN.

        Info:
            W = Width
            H = Heigth
            D = Dimension
            K = Number of kernels

        Params:
            input_dim - Input images dimension
            kernel_dim - Filters dimension. Shape (H, W, D, K)
    """
    def __init__(self, input_dim=(1200, 1920, 3), kernel_dim=(3, 3, 3, 5),
                 stride=2):
        self.params = {}
        self.input_dim = input_dim
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.padding_size = int((kernel_dim[0] - 1) / 2)

        W = tf.Variable(tf.random_normal(kernel_dim, mean=0, stddev=1e-2),
                        dtype=tf.float32)
        b = tf.Variable(tf.zeros(kernel_dim[-1]))
        self.params['W'] = W
        self.params['b'] = b
        tf.global_variables_initializer()

    def forward(self, inputs):
        """
            Executes the forward operation of the conv layer. Receives inputs,
            multiply them by the weights, following the convolution pattern,
            and sum with the biases. Applies a ReLU function at the end.
        """
        stride = [1, self.stride, self.stride, 1]
        conv_step = tf.nn.conv2d(inputs, self.params['W'], stride=stride,
                                 padding="SAME")
        return tf.nn.relu(conv_step + self.params['b'])


class Pooling_layer():
    """
        A layer that computes the max-pooling step on a CNN
    """
    def __init__(self):
        pass
