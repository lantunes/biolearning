import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from biolearning import BioLearningLayer
from sklearn.linear_model import LogisticRegression


def draw_weights(synapses, Kx, Ky):
    plt.figure(figsize=(12.9, 10))
    yy = 0
    heatmap = np.zeros((28*Ky, 28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            heatmap[y*28:(y+1)*28, x*28:(x+1)*28] = synapses[yy, :].reshape(28, 28)
            yy += 1
    nc = np.amax(np.absolute(heatmap))
    plt.axis('off')
    im = plt.imshow(heatmap, cmap='bwr', vmin=-nc, vmax=nc)
    plt.colorbar(im, ticks=[np.amin(heatmap), 0, np.amax(heatmap)])
    plt.show()

if __name__ == "__main__":

    mat = scipy.io.loadmat('mnist_all.mat')

    print("unsupervised learning...")

    num_input = 784
    training_set = np.zeros((0, num_input))
    for i in range(10):
        training_set = np.concatenate((training_set, mat['train' + str(i)]), axis=0)
    training_set = training_set / 255.0

    # test_set = np.zeros((0, num_input))
    # for i in range(10):
    #     test_set = np.concatenate((test_set, mat['test' + str(i)]), axis=0)
    # test_set = test_set / 255.0

    learning_rate = 2e-2
    num_hidden = 100
    mu = 0.0
    sigma = 1.0
    num_epochs = 200
    minibatch_size = 100
    prec = 1e-30 # controls numerical precision of the weight updates
    delta = 0.4  # Strength of the anti-hebbian learning
    p = 2.0  # Lebesgue norm of the weights
    k = 2  # ranking parameter, must be integer that is bigger or equal than 2

    layer = BioLearningLayer(num_hidden, num_input, mu, sigma)
    layer.train(M=training_set, num_epochs=num_epochs, minibatch_size=minibatch_size,
                learning_rate=learning_rate, k=k, prec=prec, delta=delta, p=p)

    # draw_weights(layer.synapses, 10, 10)

    print("constructing training set for supervised learning...")

    X = []
    Y = []

    for i in range(10):
        label = i
        examples = mat['train' + str(i)]
        examples = examples / 255.0
        for j in range(len(examples)):
            activities = layer.feedforward(examples[j])
            X.append(activities)
            Y.append(label)

    print("supervised learning...")

    reg = LogisticRegression(multi_class='multinomial', solver='sag')
    reg.fit(X, Y)

    print("done training")

    print("evaluating...")

    num_correct = 0
    n_test = 0
    for i in range(10):
        correct_label = i
        examples = mat['test' + str(i)]
        examples = examples / 255.0
        n_test += len(examples)
        for j in range(len(examples)):
            activities = layer.feedforward(examples[j])
            prediction = reg.predict([activities])
            if prediction[0] == correct_label:
                num_correct += 1

    print("%s / %s" % (num_correct, n_test))
