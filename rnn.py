import numpy as np

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, bptt_truncate=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
        self.b = np.random.normal(0, 1e-2, hidden_dim)
        self.c = np.random.normal(0, 1e-2, output_dim)
        self.U = np.random.uniform(-np.sqrt(1. / input_dim),
                                    np.sqrt(1. / input_dim),
                                    (hidden_dim, input_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                    np.sqrt(1. / hidden_dim),
                                    (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                    np.sqrt(1. / hidden_dim),
                                    (output_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        h = np.zeros((T + 1, self.hidden_dim))
        y_hat = np.zeros((T, self.output_dim))
        for t in range(T):
            a_t = np.dot(self.W, h[t-1]) + self.U[:, x[t]] + self.b
            h_t = np.tanh(a_t)
            o_t = np.dot(self.V, h_t) + self.c
            y_hat_t = np.exp(o_t - np.max(o_t)) / np.sum(np.exp(o_t - np.max(o_t)))
            # Save values of h_t and y_t
            h[t] = h_t
            y_hat[t] = y_hat_t
        return h, y_hat

    def predict(self, x):
        h, y_hat = self.forward_propagation(x)
        return np.argmax(y_hat, axis=1)

    def loss(self, x, y):
        T = len(y)
        L = 0
        h, y_hat = self.forward_propagation(x) 
        for t in range(T):
            L += -np.log(y_hat[t, y[t]])
        return L

    def loss_avg(self, X, Y):
        L = 0.
        for x, y in zip(X, Y):
            L += self.loss(x, y)
        return L / float(len(X))

    def one_hot(self, x, num_classes):
        return np.eye(num_classes)[x]

    def bptt(self, x, y):
        T = len(x)
        h, y_hat = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdb, dLdc = 0, 0
        delta__h_t_1 = np.zeros(self.hidden_dim)
        for t in np.arange(max(0, T - self.bptt_truncate), T)[::-1]:
            delta__o_t = y_hat[t]
            delta__o_t[y[t]] -= 1.
            delta__h_t = np.dot(self.V.T, delta__o_t) + \
                         np.dot(self.W.T, delta__h_t_1)
            dh_da = 1 - np.square(h[t])
            dLdV += np.outer(delta__o_t, h[t].T)
            dLdW += np.dot(np.diag(dh_da), np.outer(delta__h_t, h[t-1]))
            dLdU[:, x[t]] += dh_da * delta__h_t
            dLdc += delta__o_t
            dLdb += dh_da * delta__h_t
            delta__h_t_1 = delta__h_t
        return dLdU, dLdV, dLdW, dLdb, dLdc

    def sgd_step(self, x, y, alpha):
        dLdU, dLdV, dLdW, dLdb, dLdc = self.bptt(x, y)
        self.b -= alpha * dLdb
        self.c -= alpha * dLdc
        self.U -= alpha * dLdU
        self.V -= alpha * dLdV
        self.W -= alpha * dLdW

    def train(self, x, y, num_iter=1000, alpha=0.1):
        EXAMPLES_PER_EPOCH = len(x)
        for i in range(num_iter):
            idx = np.random.randint(len(x))
            self.sgd_step(x[idx], y[idx], alpha)
            if i % EXAMPLES_PER_EPOCH == 0:
                print("Loss @{}: {}".format(i, self.loss_avg(x, y)))

    def gradient_check(self, x, y, eps=1e-10):
        dLdU, dLdV, dLdW, dLdb, dLdc = self.bptt(x, y)
        # Compute approximate gradients for U, V, W
        b_grad = np.zeros(self.b.shape)
        c_grad = np.zeros(self.c.shape)
        U_grad = np.zeros(self.U.shape)
        V_grad = np.zeros(self.V.shape)
        W_grad = np.zeros(self.W.shape)
        for i in range(b_grad.shape[0]):
            self.b[i] -= eps
            L1 = self.loss(x, y)
            self.b[i] += 2 * eps
            L2 = self.loss(x, y)
            self.b[i] -= eps
            b_grad[i] = (L2 - L1) / (2 * eps)
        for i in range(c_grad.shape[0]):
            self.c[i] -= eps
            L1 = self.loss(x, y)
            self.c[i] += 2 * eps
            L2 = self.loss(x, y)
            self.c[i] -= eps
            c_grad[i] = (L2 - L1) / (2 * eps)
        for i in range(U_grad.shape[0]):
            for j in range(U_grad.shape[1]):
                self.U[i, j] -= eps
                L1 = self.loss(x, y)
                self.U[i, j] += 2 * eps
                L2 = self.loss(x, y)
                self.U[i, j] -= eps
                U_grad[i, j] = (L2 - L1) / (2 * eps)
        for i in range(V_grad.shape[0]):
            for j in range(V_grad.shape[1]):
                self.V[i, j] -= eps
                L1 = self.loss(x, y)
                self.V[i, j] += 2 * eps
                L2 = self.loss(x, y)
                self.V[i, j] -= eps
                V_grad[i, j] = (L2 - L1) / (2 * eps)
        for i in range(W_grad.shape[0]):
            for j in range(W_grad.shape[1]):
                self.W[i, j] -= eps
                L1 = self.loss(x, y)
                self.W[i, j] += 2 * eps
                L2 = self.loss(x, y)
                self.W[i, j] -= eps
                W_grad[i, j] = (L2 - L1) / (2 * eps)
        print('b gradient MSE:', np.sum(np.square(dLdb - b_grad)))
        print('c gradient MSE:', np.sum(np.square(dLdc - c_grad)))
        print('U gradient MSE:', np.sum(np.square(dLdU - U_grad)))
        print('V gradient MSE:', np.sum(np.square(dLdV - V_grad)))
        print('W gradient MSE:', np.sum(np.square(dLdW - W_grad)))
