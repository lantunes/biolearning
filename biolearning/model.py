import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class BioLearningLayer:
    def __init__(self, num_hidden, num_input, mu=0.0, sigma=1.0):
        self._num_hidden = num_hidden
        self._num_input = num_input
        self._synapses = np.random.normal(mu, sigma, (self._num_hidden, num_input))

    def train(self, M, num_epochs, minibatch_size, learning_rate=2e-2, k=2, prec=1e-30, delta=0.4, p=2.0):
        for nep in range(num_epochs):
            logging.info("epoch: %s" % (nep+1))
            annealed_learning_rate = learning_rate * (1 - nep / num_epochs)
            Ns = len(M)
            M = M[np.random.permutation(Ns), :]
            for i in range(Ns // minibatch_size):
                inputs = np.transpose(M[i * minibatch_size:(i + 1) * minibatch_size, :])
                sig = np.sign(self._synapses)
                tot_input = np.dot(sig*np.absolute(self._synapses)**(p-1), inputs)

                y = np.argsort(tot_input, axis=0)
                yl = np.zeros((self._num_hidden, minibatch_size))
                yl[y[self._num_hidden - 1, :], np.arange(minibatch_size)] = 1.0
                yl[y[self._num_hidden - k], np.arange(minibatch_size)] = -delta

                xx = np.sum(np.multiply(yl, tot_input), 1)
                ds = np.dot(yl, np.transpose(inputs)) - np.multiply(np.tile(xx.reshape(xx.shape[0], 1), (1, self._num_input)),
                                                                    self._synapses)

                nc = np.amax(np.absolute(ds))
                if nc < prec:
                    nc = prec
                self._synapses += annealed_learning_rate * np.true_divide(ds, nc)

    @property
    def synapses(self):
        return self._synapses

    def feedforward(self, input):
        return self._relu(np.matmul(input.reshape(1, input.shape[0]), np.transpose(self._synapses)))[0]

    def _relu(self, x):
        return np.maximum(0, x)
