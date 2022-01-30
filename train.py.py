import numpy as np
import matplotlib.pyplot as plt

X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]

lr = 0.01

layer_depth = {
    'layer1': 2,
    'output': 1,
}

def mse(y_true, y_pred):
    return 1 / 1 * (y_true - y_pred) ** 2
# passes_in_input_l = X[1]
#
# np.random.seed(45)
# w_layer1 = 0.01 * np.random.rand(1, layer_depth['layer1'])
# w1, w2 = w_layer1[0]
#
# bias_layer1 = 0.1 * np.random.rand(1, layer_depth['layer1'])
# b1, b2 = bias_layer1[0]
#
# # forward prop..
#
# l1_out = w1 * passes_in_input_l + b1
# l2_out = w2 * passes_in_input_l + b2
#
# w_output = 0.01 * np.random.rand(2, layer_depth['output'])
# w3, w4 = w_output
#
# bias_output = 0.1 * np.random.rand(1, 1)
# b3 = bias_output[0][0]
#
# output_out = (w3 * l1_out + w4 * l2_out) + b3
#
# output = output_out[0]
#
#
#
#
# print('loss :', mse(4, output))
#
# # back prop.....
#
# de_dout = -2 * (4 - output)
# dout_dw3 = l1_out
# de_dw3 = de_dout * dout_dw3
# de_db3 = de_dout * 1
#
# prev_w3 = w3
# w3 = w3 - (lr * de_dw3)
# b3 = b3 - (lr * de_db3)
#
# dout_dw4 = l2_out
# de_dw4 = de_dout * dout_dw4
#
# prev_w4 = w4
# w4 = w4 - (lr * de_dw4)
#
# dout_dl1 = prev_w3
# dl1_dw1 = passes_in_input_l
# de_dw1 = de_dout * dout_dl1 * dl1_dw1
# de_db1 = de_dout * dout_dl1 * 1
#
# prev_w1 = w1
# w1 = prev_w1 - (lr * de_dw1)
# b1 = b1 - (lr * de_db1)
#
# dout_dl2 = l2_out
# dl2_dw2 = passes_in_input_l
#
# de_dw2 = de_dout * dout_dl2 * dl2_dw2
# de_db2 = de_dout * dout_dl2 * 1
#
# prev_w2 = w2
# w2 = prev_w2 - (lr * de_dw2)
# b2 = b2 - (lr * de_db2)
#
# # forward prop .... iter - 2
#
# l1_out = w1 * passes_in_input_l + b1
# l2_out = w2 * passes_in_input_l + b2
#
# output_out = (w3 * l1_out + w4 * l2_out) + b3
# output = output_out
#
# loss = mse(4, output)
# print(loss[0])
#
# # back prop....
#
# de_dout = -2 * (4 - output)
# dout_dw3 = l1_out
# de_dw3 = de_dout * dout_dw3
# de_db3 = de_dout * 1
#
# prev_w3 = w3
# w3 = w3 - (lr * de_dw3)
# b3 = b3 - (lr * de_db3)
#
# dout_dw4 = l2_out
# de_dw4 = de_dout * dout_dw4
#
# prev_w4 = w4
# w4 = w4 - (lr * de_dw4)
#
# dout_dl1 = prev_w3
# dl1_dw1 = passes_in_input_l
# de_dw1 = de_dout * dout_dl1 * dl1_dw1
# de_db1 = de_dout * dout_dl1 * 1
#
# prev_w1 = w1
# w1 = prev_w1 - (lr * de_dw1)
# b1 = b1 - (lr * de_db1)
#
# dout_dl2 = l2_out
# dl2_dw2 = passes_in_input_l
#
# de_dw2 = de_dout * dout_dl2 * dl2_dw2
# de_db2 = de_dout * dout_dl2 * 1
#
# prev_w2 = w2
# w2 = prev_w2 - (lr * de_dw2)
# b2 = b2 - (lr * de_db2)
#
# # iter - 2:
# # forward- prop....
#
# l1_out = w1 * passes_in_input_l + b1
# l2_out = w2 * passes_in_input_l + b2
#
# output_out = (w3 * l1_out + w4 * l2_out) + b3
# output = output_out
#
# loss = mse(4, output)
# print(loss[0])
#
# #....
passes_in_input_l = X[1]

np.random.seed(45)
w_layer1 = 0.01 * np.random.rand(1, layer_depth['layer1'])
w1, w2 = w_layer1[0]

bias_layer1 = 0.1 * np.random.rand(1, layer_depth['layer1'])
b1, b2 = bias_layer1[0]

w_output = 0.01 * np.random.rand(2, layer_depth['output'])
w3, w4 = w_output

bias_output = 0.1 * np.random.rand(1, 1)
b3 = bias_output[0][0]

W = []
B = []
o = []
for i in range(20):
    # forward prop...
    l1_out = w1 * passes_in_input_l + b1
    l2_out = w2 * passes_in_input_l + b2

    output_out = (w3 * l1_out + w4 * l2_out) + b3
    output = output_out

    loss = mse(4, output)
    print(loss[0])
    # back prop....

    de_dout = -2 * (4 - output)
    dout_dw3 = l1_out
    de_dw3 = de_dout * dout_dw3
    de_db3 = de_dout * 1

    prev_w3 = w3
    w3 = w3 - (lr * de_dw3)
    b3 = b3 - (lr * de_db3)

    dout_dw4 = l2_out
    de_dw4 = de_dout * dout_dw4

    prev_w4 = w4
    w4 = w4 - (lr * de_dw4)

    dout_dl1 = prev_w3
    dl1_dw1 = passes_in_input_l
    de_dw1 = de_dout * dout_dl1 * dl1_dw1
    de_db1 = de_dout * dout_dl1 * 1

    prev_w1 = w1
    w1 = prev_w1 - (lr * de_dw1)
    b1 = b1 - (lr * de_db1)

    dout_dl2 = l2_out
    dl2_dw2 = passes_in_input_l

    de_dw2 = de_dout * dout_dl2 * dl2_dw2
    de_db2 = de_dout * dout_dl2 * 1

    prev_w2 = w2
    w2 = prev_w2 - (lr * de_dw2)
    b2 = b2 - (lr * de_db2)

    # forward prop...
    l1_out = w1 * passes_in_input_l + b1
    l2_out = w2 * passes_in_input_l + b2

    output_out = (w3 * l1_out + w4 * l2_out) + b3
    output = output_out
    print(output)
    o.append(output[0])
    loss = mse(4, output)
    print(loss[0])
    W = [w1, w2, w3, w4]
    B = [b1, b2, b3]


print(W, B)
