from libraries import *
import tensorflow as tf # for matmul only

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]


# create your network

layer1 = Dense(1, 2)
layer2 = Dense(2, 2)
outputs = Dense(2, 1)
Activation_layer = Relu()

LEARNING_RATE = 0.001 #lr
NETWORK_DEPTH = 3 # network of three layers

layer = [layer1, layer2, outputs] # should specify the layers are in use


def back_prp(layer, epochs):
    for i in range(epochs):
        in_layer = NETWORK_DEPTH - 1
        layer1.forward_propagation(x[0])
        layer2.forward_propagation(layer1.output)
        outputs.forward_propagation(layer2.output)
        Activation_layer.forward(outputs.output)
        loss = MSE(labels=[y[0]], output=Activation_layer.output)
        loss.back_prop()
        print(loss.e)
        Activation_layer.back_prop()
        de_dA = loss.e
        dA_do = Activation_layer.e

        for i in range(NETWORK_DEPTH):
            if in_layer == NETWORK_DEPTH - 1:
                layer[in_layer].back_prop()
                de_dA = loss.e
                dA_do = Activation_layer.e
                do_dw = np.array([layer[in_layer].e])
                set_up = de_dA * dA_do
                set_up = np.array(set_up)
                de_dw = tf.matmul(set_up, do_dw)
                set_up = LEARNING_RATE * de_dw
                set_up = set_up[0]
                W = layer[in_layer].weights[0]
                sum_ = add(W, set_up)
                print(W.shape, sum_.shape)
                layer[in_layer].weights = np.array(sum_).reshape(2, 1)
                print(f'layer{in_layer} back propagated')
                in_layer = in_layer - 1
            else:
                layer[in_layer].back_prop()
                de_dA = loss.e
                dA_do = Activation_layer.e
                previous_layer_weights = []
                for l in layer[in_layer:]:
                    previous_layer_weights.append(l.weights)
                do_dw = de_dA * dA_do
                lay = layer[in_layer].weights
                fin = []
                for i in layer[in_layer + 1:]:
                    fin = np.dot(lay, i.weights)
                de_dw = tf.matmul(lay, fin)
                set_up = LEARNING_RATE * de_dw
                wei = add(layer[in_layer].weights[0], set_up)
                print(f'weights of {layer[in_layer].weights} is setted to {wei}')
                layer[in_layer].weights = wei
                print(f'layer{in_layer} back propagated')
                in_layer = in_layer - 1


def add(A, B):
    r = np.zeros(B.shape)
    try:
        for i in range(len(A)):
            r[i] = A[i] + B[i]
    except:
        A = list(A)
        A.append(0)
        add(A, B)
    return r


back_prp(layer, epochs=10)


'''

***  you have some knowledge to run this  ***

'''