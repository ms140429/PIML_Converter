"""
==================================================================================================
Author: Shuai Zhao @ Aalborg University, szh@energy.aau.dk
Note:
     * The code and data accompany the paper:
       S. Zhao, Y. Peng, Y. Zhang, and H. Wang, "Parameter Estimation of Power Electronic
       Converters with Physics-informed Machine Learning", IEEE Trans. Power Electronics, 2022

     * The Tensorflow verion is 1.15.0. The code is developed based on the github repository
       https://github.com/maziarraissi/PINNs, where more useful information can be found therein.
==================================================================================================
"""

import sys
import tensorflow as tf
import numpy as np
import time
import scipy.io

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, q, splitIdx1, splitIdx2, splitIdx3, otherParams):

        self.lb = lb
        self.ub = ub

        self.x0 = x0
        self.x1 = x1

        self.u0 = u0
        self.u1 = u1

        self.layers = layers
        self.dt = dt
        self.q = max(q, 1)

        self.splitIdx1 = splitIdx1
        self.splitIdx2 = splitIdx2
        self.splitIdx3 = splitIdx3

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        self.otherParams = otherParams

        self.L = tf.Variable([2], dtype=tf.float32)
        self.RL = tf.Variable([0.039], dtype=tf.float32)
        self.C = tf.Variable([0.412], dtype=tf.float32)
        self.RC = tf.Variable([1.59], dtype=tf.float32)
        self.Rdson = tf.Variable([1.22], dtype=tf.float32)
        self.Rload1 = tf.Variable([1.22], dtype=tf.float32)
        self.Rload2 = tf.Variable([1.22], dtype=tf.float32)
        self.Rload3 = tf.Variable([1.22], dtype=tf.float32)
        self.vIn = tf.Variable([0.87], dtype=tf.float32)
        self.vF = tf.Variable([0.1], dtype=tf.float32)

        tmp = np.float32(np.loadtxt(
            'Butcher_tableau/Butcher_IRK%d.txt' % (q),
            ndmin=2))
        weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))
        self.IRK_alpha = weights[0:-1, :]
        self.IRK_beta = weights[-1:, :]
        self.IRK_times = tmp[q ** 2 + q:]

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x0_tf = tf.placeholder(tf.float32, shape=(None, self.x0.shape[1]))
        self.x1_tf = tf.placeholder(tf.float32, shape=(None, self.x1.shape[1]))
        self.u0_tf = tf.placeholder(tf.float32, shape=(None, self.u0.shape[1]))
        self.u1_tf = tf.placeholder(tf.float32, shape=(None, self.u1.shape[1]))

        self.U0_pred, self.V0_pred = self.net_U0(self.x0_tf)  # 2*N0 x q
        self.U1_pred, self.V1_pred = self.net_U1(self.x1_tf)  # 2*N1 x q

        self.loss = tf.reduce_sum(tf.square(self.u0_tf[:, 1:2] - self.V0_pred)) + \
                    tf.reduce_sum(tf.square(self.u1_tf[:, 1:2] - self.V1_pred)) + \
                    tf.reduce_sum(tf.square(self.u0_tf[:, 0:1] - self.U0_pred)) + \
                    tf.reduce_sum(tf.square(self.u1_tf[:, 0:1] - self.U1_pred))  # + \

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'iprint': 0,
                                                                         'maxiter': otherParams.LBFGSEpoch,
                                                                         'maxfun': otherParams.LBFGSEpoch,
                                                                         'maxcor': 50,
                                                                         'maxls': 100,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers): # initialize a fully-connected neural network
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_U0(self, x):  # establish the backward relationship eq. (4)

        L = tf.exp(self.L) * 1e-4
        RL = tf.exp(self.RL) * 1e-1
        C = tf.exp(self.C) * 1e-4
        RC = tf.exp(self.RC) * 1e-1
        Rdson = tf.exp(self.Rdson) * 1e-1
        Rload1 = tf.exp(self.Rload1)
        Rload2 = tf.exp(self.Rload2)
        Rload3 = tf.exp(self.Rload3)
        vIn = tf.exp(self.vIn) * 1e1
        vF = tf.exp(self.vF)

        y = x[:, 2:3]
        yOff = x[:, 3:4]
        dt = x[:, 4:5]

        Rload = tf.concat((tf.ones((self.splitIdx1 * 2, 1)) * Rload1, tf.ones((self.splitIdx2 * 2, 1)) * Rload2,
                           tf.ones((self.splitIdx3 * 2, 1)) * Rload3), 0)

        current_and_voltage = self.neural_net(x, self.weights, self.biases)
        u = current_and_voltage[:, 0:self.q]
        v = current_and_voltage[:, self.q:self.q * 2]

        F_u = -((y * (RL + Rdson)) * u + (yOff * (RL)) * u + v - y * vIn + yOff * vF) / L  # inductor current model
        U0 = u - dt * tf.matmul(F_u, self.IRK_alpha.T)
        F_v = (C * RC * Rload * (F_u) + Rload * u - v) / (C * (RC + Rload))  # output voltage model
        V0 = v - dt * tf.matmul(F_v, self.IRK_alpha.T)

        return U0, V0

    def net_U1(self, x):  # establish the forward relationship eq. (7)

        L = tf.exp(self.L) * 1e-4
        RL = tf.exp(self.RL) * 1e-1
        C = tf.exp(self.C) * 1e-4
        RC = tf.exp(self.RC) * 1e-1
        Rdson = tf.exp(self.Rdson) * 1e-1
        Rload1 = tf.exp(self.Rload1)
        Rload2 = tf.exp(self.Rload2)
        Rload3 = tf.exp(self.Rload3)
        vIn = tf.exp(self.vIn) * 1e1
        vF = tf.exp(self.vF)

        y = x[:, 2:3]
        yOff = x[:, 3:4]
        dt = x[:, 4:5]

        Rload = tf.concat((tf.ones((self.splitIdx1 * 2, 1)) * Rload1, tf.ones((self.splitIdx2 * 2, 1)) * Rload2,
                           tf.ones((self.splitIdx3 * 2, 1)) * Rload3), 0)

        current_and_voltage = self.neural_net(x, self.weights, self.biases)
        u = current_and_voltage[:, 0:self.q]
        v = current_and_voltage[:, self.q:self.q * 2]

        F_u = -((y * (RL + Rdson)) * u + (yOff * (RL)) * u + v - y * vIn + yOff * vF) / L  # inductor current model
        U1 = u + dt * tf.matmul(F_u, (self.IRK_beta - self.IRK_alpha).T)
        F_v = (C * RC * Rload * (F_u) + Rload * u - v) / (C * (RC + Rload))  # output voltage model
        V1 = v + dt * tf.matmul(F_v, (self.IRK_beta - self.IRK_alpha).T)

        return U1, V1

    def callback(self, loss, L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vIn, vF):
        L_value = np.abs(np.exp(L) / self.otherParams.nL * 100 - 100)
        RL_value = np.abs(np.exp(RL) / self.otherParams.nRL * 100 - 100)
        C_value = np.abs(np.exp(C) / self.otherParams.nC * 100 - 100)
        RC_value = np.abs(np.exp(RC) / self.otherParams.nRC * 100 - 100)
        Rdson_value = np.abs(np.exp(Rdson) / self.otherParams.nRdson * 100 - 100)
        Rload1_value = np.abs(np.exp(Rload1) / self.otherParams.nRload1 * 100 - 100)
        Rload2_value = np.abs(np.exp(Rload2) / self.otherParams.nRload2 * 100 - 100)
        Rload3_value = np.abs(np.exp(Rload3) / self.otherParams.nRload3 * 100 - 100)
        vIn_value = np.abs(np.exp(vIn) / self.otherParams.nVin * 100 - 100)
        vF_value = np.abs(np.exp(vF) / self.otherParams.nVF * 100 - 100)
        meanError = np.mean([L_value, RL_value, C_value,
                             RC_value, Rdson_value, Rload1_value,
                             Rload2_value, Rload3_value,
                             vIn_value, vF_value])

        print(
            'Loss: %.3e, mean: %.1f,  L: %.1f, RL: %.1f, C: %.1f, RC: %.1f, Rdson: %.1f, Rload1: %.1f, Rload2: %.1f, Rload3: %.1f, vIn: %.1f, vF: %.1f' % (
                loss, meanError, L_value, RL_value,
                C_value, RC_value,
                Rdson_value,
                Rload1_value, Rload2_value,
                Rload3_value,
                vIn_value, vF_value))

    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0, self.u0_tf: self.u0,
                   self.x1_tf: self.x1, self.u1_tf: self.u1}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print training info
            if it % 10000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                L = np.abs(np.exp(self.sess.run(self.L)) / self.otherParams.nL * 100 - 100)
                RL = np.abs(np.exp(self.sess.run(self.RL)) / self.otherParams.nRL * 100 - 100)
                C = np.abs(np.exp(self.sess.run(self.C)) / self.otherParams.nC * 100 - 100)
                RC = np.abs(np.exp(self.sess.run(self.RC)) / self.otherParams.nRC * 100 - 100)
                Rdson = np.abs(np.exp(self.sess.run(self.Rdson)) / self.otherParams.nRdson * 100 - 100)
                Rload1 = np.abs(np.exp(self.sess.run(self.Rload1)) / self.otherParams.nRload1 * 100 - 100)
                Rload2 = np.abs(np.exp(self.sess.run(self.Rload2)) / self.otherParams.nRload2 * 100 - 100)
                Rload3 = np.abs(np.exp(self.sess.run(self.Rload3)) / self.otherParams.nRload3 * 100 - 100)
                vIn = np.abs(np.exp(self.sess.run(self.vIn)) / self.otherParams.nVin * 100 - 100)
                vF = np.abs(np.exp(self.sess.run(self.vF)) / self.otherParams.nVF * 100 - 100)
                meanError = np.mean([L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vIn, vF])

                print(
                    'Iterations: %d, Loss: %.3e, mean: %.1e, L: %.1f, RL: %.1f, C: %.1f, RC: %.1f, Rdson: %.1f, Rload1: %.1f, Rload2: %.1f, Rload3: %.1f, vIn: %.1f, vF: %.1f, Time: %.2f' %
                    (it, loss_value, meanError, L, RL, C, RC, Rdson, Rload1, Rload2, Rload3, vIn, vF, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.L, self.RL, self.C, self.RC, self.Rdson, self.Rload1,
                                         self.Rload2, self.Rload3, self.vIn, self.vF],
                                loss_callback=self.callback)

    def predict(self, x_star_Backward, x_star_Forward, dt):

        U0_star = self.sess.run(self.U0_pred,
                                {self.x0_tf: x_star_Backward})
        V0_star = self.sess.run(self.V0_pred,
                                {self.x0_tf: x_star_Backward})
        U1_star = self.sess.run(self.U1_pred,
                                {self.x1_tf: x_star_Forward})
        V1_star = self.sess.run(self.V1_pred,
                                {self.x1_tf: x_star_Forward})

        return U0_star, V0_star, U1_star, V1_star


class otherParams: # simulation settings
    def __init__(self):
        self.AdamEpoch = 200000
        self.AdamLearningRate = 1e-3
        self.LBFGSEpoch = 50000
        self.nL = 7.25
        self.nRL = 3.14
        self.nC = 1.645
        self.nRC = 2.01
        self.nRdson = 2.21
        self.nRload1 = 3.1
        self.nRload2 = 10.2
        self.nRload3 = 6.1
        self.nVin = 4.8
        self.nVF = 1


if __name__ == "__main__":

    for testIdx in range(0, 7, 1): # loop to test all 7 cases automatically
        fileNameAutoGen = 'Simulation_data/buckSimulation_%s.mat' % testIdx
        data = scipy.io.loadmat(fileNameAutoGen)
        xCurrent = data['CurrentInput']
        xVoltage = data['VoltageInput']
        xSwitch = data['Dswitch']
        yCurrent = data['Current']
        yVoltage = data['Voltage']
        Indicator = data['forwaredBackwaredIndicator']
        dt = data['dt']

        idx_forward = np.transpose(np.nonzero(Indicator[:, 0:1] == -2))
        idx_forward = idx_forward[:, 0:1]
        idx_backward = np.transpose(np.nonzero(Indicator[:, 0:1] == 2))
        idx_backward = idx_backward[:, 0:1]

        q = 20
        layers = [5, 50, 50, 50, 50, 50, q * 2] # architecture of the neural network
        X = np.concatenate([xCurrent, xVoltage, xSwitch, 1 - xSwitch, dt], 1)
        lb = X.min(0)
        ub = X.max(0)

        x0 = np.concatenate(
            [xCurrent[idx_forward], xVoltage[idx_forward], xSwitch[idx_forward], 1 - xSwitch[idx_forward],
             dt[idx_forward]], 1)
        x0 = np.reshape(x0, (-1, 5))
        u0 = np.concatenate([yCurrent[idx_forward], yVoltage[idx_forward]], 1)
        u0 = np.reshape(u0, (-1, 2))

        x1 = np.concatenate(
            [xCurrent[idx_backward], xVoltage[idx_backward], xSwitch[idx_backward], 1 - xSwitch[idx_backward],
             dt[idx_backward]], 1)
        x1 = np.reshape(x1, (-1, 5))
        u1 = np.concatenate([yCurrent[idx_backward], yVoltage[idx_backward]], 1)
        u1 = np.reshape(u1, (-1, 2))

        # the information of number of data points in each trasient regimes.
        splitIdx1 = 120  # otherwise it will be 1 for no data case of the signal transient
        splitIdx2 = 120
        splitIdx3 = 120
        Params = otherParams()

        model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q, splitIdx1, splitIdx2, splitIdx3, Params)
        model.train(Params.AdamEpoch)

        L = np.exp(model.sess.run(model.L))
        RL = np.exp(model.sess.run(model.RL))
        C = np.exp(model.sess.run(model.C))
        RC = np.exp(model.sess.run(model.RC))
        Rdson = np.exp(model.sess.run(model.Rdson))
        Rload1 = np.exp(model.sess.run(model.Rload1))
        Rload2 = np.exp(model.sess.run(model.Rload2))
        Rload3 = np.exp(model.sess.run(model.Rload3))
        vIn = np.exp(model.sess.run(model.vIn))
        vF = np.exp(model.sess.run(model.vF))
        RLRdson = RL + Rdson

        # Norminal values of the simulation setting, please refer to Table I in the paper.
        L_value = np.abs(np.mean(L) / Params.nL * 100 - 100)
        RL_value = np.abs(np.mean(RL) / Params.nRL * 100 - 100)
        C_value = np.abs(np.mean(C) / Params.nC * 100 - 100)
        RC_value = np.abs(np.mean(RC) / Params.nRC * 100 - 100)
        Rdson_value = np.abs(np.mean(Rdson) / Params.nRdson * 100 - 100)
        Rload1_value = np.abs(np.mean(Rload1) / Params.nRload1 * 100 - 100)
        Rload2_value = np.abs(np.mean(Rload2) / Params.nRload2 * 100 - 100)
        Rload3_value = np.abs(np.mean(Rload3) / Params.nRload3 * 100 - 100)
        vIn_value = np.abs(np.mean(vIn) / Params.nVin * 100 - 100)
        vF_value = np.abs(np.mean(vF) / Params.nVF * 100 - 100)

        meanError = np.mean([L_value, RL_value, C_value,
                             RC_value, Rdson_value, Rload1_value,
                             Rload2_value, Rload3_value, vIn_value, vF_value])

        text_file = open("Result.txt", "a+")
        text_file.write(
            "buckSimulation_%s: mean:%.2f, L: %.2f, RL: %.2f, C: %.2f, RC: %.2f, Rdson: %.2f, Rload1: %.2f, Rload2: %.2f, Rload3: %.2f, vIn: %.2f, vF: %.2f\n" % (
                testIdx, meanError, L_value, RL_value, C_value, RC_value, Rdson_value, Rload1_value,
                Rload2_value,
                Rload3_value, vIn_value, vF_value))
        text_file.close()
