import numpy as np

def dtanh(x):
    return 1 - np.square(x)

def dsigmoid(x):
    return x * (1 - x)

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Input weights
        self.b = self.init_bias_term(hidden_dim)
        self.U = self.init_weight_matrix(hidden_dim, input_dim)
        self.W = self.init_weight_matrix(hidden_dim, hidden_dim)
        # Input gate weights 
        self.bg = self.init_bias_term(hidden_dim)
        self.Ug = self.init_weight_matrix(hidden_dim, input_dim)
        self.Wg = self.init_weight_matrix(hidden_dim, hidden_dim)
        # Forget gate weights 
        self.bf = self.init_bias_term(hidden_dim)
        self.Uf = self.init_weight_matrix(hidden_dim, input_dim)
        self.Wf = self.init_weight_matrix(hidden_dim, hidden_dim)
        # Output gate weights 
        self.bo = self.init_bias_term(hidden_dim)
        self.Uo = self.init_weight_matrix(hidden_dim, input_dim)
        self.Wo = self.init_weight_matrix(hidden_dim, hidden_dim)
        # Output weights
        self.by = self.init_bias_term(output_dim)
        self.Wy = self.init_weight_matrix(output_dim, hidden_dim)

    def init_weight_matrix(self, output_dim, input_dim):
        shape = (output_dim, input_dim)
        return np.random.uniform(
                -np.sqrt(1. / input_dim), np.sqrt(1. / input_dim), shape)

    def init_bias_term(self, input_dim):
        return np.random.normal(0, 1e-2, input_dim)

    def forward_propagation(self, x):
        sigmoid = self.sigmoid
        softmax = self.softmax
        T = len(x)
        h = np.zeros((T + 1, self.hidden_dim))
        s = np.zeros((T + 1, self.hidden_dim))
        # gates
        f = np.zeros((T, self.hidden_dim))
        g = np.zeros((T, self.hidden_dim))
        q = np.zeros((T, self.hidden_dim))
        # input + outputs
        x_inputs = np.zeros((T, self.hidden_dim))
        y_hat = np.zeros((T, self.output_dim))
        for t in range(T):
            h_old = h[t-1]
            x_inputs[t] = sigmoid(self.b + self.U[:, x[t]] \
                                         + np.dot(self.W, h_old))
            f[t] = sigmoid(self.bf + self.Uf[:, x[t]] \
                                   + np.dot(self.Wf, h_old))
            g[t] = sigmoid(self.bg + self.Ug[:, x[t]] \
                                   + np.dot(self.Wg, h_old))
            q[t] = sigmoid(self.bo + self.Uo[:, x[t]] \
                                   + np.dot(self.Wo, h_old))
            s[t] = f[t] * s[t-1] + g[t] * x_inputs[t]
            h[t] = q[t] * np.tanh(s[t])
            y_hat[t] = softmax(self.by + np.dot(self.Wy, h[t]))
        hidden_states = [h, s]
        gates = [f, g, q]
        return x_inputs, hidden_states, gates, y_hat

    # Forward propagation but only output probs
    def forward_probs(self, x):
        x_inputs, hidden_states, gates, y_hat = self.forward_propagation(x)
        return y_hat

    def bptt(self, x, y):
        T = len(x)
        x_inputs, hidden_states, gates, y_hat = self.forward_propagation(x)
        h, s = hidden_states
        f, g, q = gates
        # Initialize gradient vars
        db, dU, dW = 0, np.zeros(self.U.shape), np.zeros(self.W.shape)
        dbf, dUf, dWf = 0, np.zeros(self.Uf.shape), np.zeros(self.Wf.shape)
        dbg, dUg, dWg = 0, np.zeros(self.Ug.shape), np.zeros(self.Wg.shape)
        dbo, dUo, dWo = 0, np.zeros(self.Uo.shape), np.zeros(self.Wo.shape)
        dby, dWy = 0, np.zeros(self.Wy.shape)
        # Retain h_t and s_t derivatives over time
        delta__h_t_1 = np.zeros(self.hidden_dim)
        delta__s_t_1 = np.zeros(self.hidden_dim)
        for t in np.arange(0, T)[::-1]:
            # Compute deltas
            delta__o_t = y_hat[t]
            delta__o_t[y[t]] -= 1.
            delta__h_t = np.dot(self.Wy.T, delta__o_t) + delta__h_t_1
            delta__q_t = np.tanh(s[t]) * delta__h_t * dsigmoid(q[t])
            delta__s_t = q[t] * delta__h_t * dtanh(np.tanh(s[t])) + delta__s_t_1
            delta__f_t = s[t-1] * delta__s_t * dsigmoid(f[t])
            delta__g_t = x_inputs[t] * delta__s_t * dsigmoid(g[t])
            delta__i_t = g[t] * delta__s_t * dsigmoid(x_inputs[t])
            # Compute and apply gradients
            dby += delta__o_t
            dWy += np.outer(delta__o_t, h[t].T)
            dbf += delta__f_t
            dUf[:, x[t]] += delta__f_t
            dWf += np.outer(delta__f_t, h[t-1].T)
            dbg += delta__g_t
            dUg[:, x[t]] += delta__g_t
            dWg += np.outer(delta__g_t, h[t-1].T)
            dbo += delta__q_t
            dUo[:, x[t]] += delta__q_t
            dWo += np.outer(delta__q_t, h[t-1].T)
            db  += delta__i_t
            dU[:, x[t]]  += delta__i_t
            dW  += np.outer(delta__i_t, h[t-1].T)
            # Compute dX's in order to compute delta__h_t_1
            dXf = np.dot(self.Wf.T, delta__f_t)
            dXg = np.dot(self.Wg.T, delta__g_t)
            dXq = np.dot(self.Wo.T, delta__q_t)
            dXs = np.dot(self.W.T,  delta__i_t)
            # Set h_t and s_t derivatives
            delta__h_t_1 = dXf + dXg + dXq + dXs
            delta__s_t_1 = delta__s_t * f[t]
        # Record gradients in dictionary
        grad_dict = {
            'b': db, 'U': dU, 'W': dW,
            'bf': dbf, 'Uf': dUf, 'Wf': dWf,
            'bg': dbg, 'Ug': dUg, 'Wg': dWg,
            'bo': dbo, 'Uo': dUo, 'Wo': dWo,
            'by': dby, 'Wy': dWy
        }
        return grad_dict

    def predict(self, x):
        x_inputs, hidden_states, gates, y_hat = self.forward_propagation(x)
        return np.argmax(y_hat, axis=1)

    def loss(self, x, y):
        T = len(y)
        L = 0
        x_inputs, hidden_states, gates, y_hat = self.forward_propagation(x) 
        for t in range(T):
            L += -np.log(y_hat[t, y[t]])
        return L

    def loss_avg(self, X, Y):
        L = 0.
        for x, y in zip(X, Y):
            L += self.loss(x, y)
        return L / float(len(X))

    def sgd_step(self, x, y, weights_dict, alpha):
        grad_dict = self.bptt(x, y)
        for weight_name, gradient in grad_dict.iteritems():
            weights_dict[weight_name] -= alpha * gradient

    def train(self, x, y, num_iter=1000, alpha=0.1):
        EXAMPLES_PER_EPOCH = len(x)
        weights_dict = self.get_weights_dict()
        for i in range(num_iter):
            idx = np.random.randint(len(x))
            self.sgd_step(x[idx], y[idx], weights_dict, alpha)
            if i % EXAMPLES_PER_EPOCH == 0:
                # print("Loss @{}: {}".format(i, self.loss_avg(x, y)))
                idx = np.random.choice(np.arange(len(x)), int(0.1 * len(x)), False)
                x_sample, y_sample = np.array(x)[idx], np.array(y)[idx]
                print("Loss @{}: {}".format(i, self.loss_avg(x_sample, y_sample)))

    def gradient_check_one_elem(self, x, y, grad_dict,
                                weights_dict, key, eps=1e-10):
        weights_matrix = weights_dict[key]
        approx_grad = np.zeros(weights_matrix.shape)
        if len(weights_matrix.shape) == 1:
            for i in range(len(weights_matrix)):
                weights_matrix[i] += eps
                L1 = self.loss(x, y)
                weights_matrix[i] += 2 * eps
                L2 = self.loss(x, y)
                weights_matrix[i] -= eps
                approx_grad[i] = (L2 - L1) / (2 * eps)
        elif len(weights_matrix.shape) == 2:
            for i in range(weights_matrix.shape[0]):
                for j in range(weights_matrix.shape[1]):
                    weights_matrix[i, j] += eps
                    L1 = self.loss(x, y)
                    weights_matrix[i, j] += 2 * eps
                    L2 = self.loss(x, y)
                    weights_matrix[i, j] -= eps
                    approx_grad[i, j] = (L2 - L1) / (2 * eps)
        print('{} gradient MSE: {}'.format(
              key, np.sum(np.square(approx_grad - grad_dict[key]))))


    def gradient_check(self, x, y, eps=1e-10):
        weights_dict = self.get_weights_dict()
        grad_dict = self.bptt(x, y)
        for weight in weights_dict: 
            self.gradient_check_one_elem(x, y, grad_dict,
                                         weights_dict, weight, eps)
    def get_weights_dict(self):
        weights_dict = {
            'b': self.b, 'U': self.U, 'W': self.W,
            'bf': self.bf, 'Uf': self.Uf, 'Wf': self.Wf,
            'bg': self.bg, 'Ug': self.Ug, 'Wg': self.Wg,
            'bo': self.bo, 'Uo': self.Uo, 'Wo': self.Wo,
            'by': self.by, 'Wy': self.Wy
        }
        return weights_dict

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
