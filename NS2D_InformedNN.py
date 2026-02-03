
from  NeuralNetworkSetup import NeuralNetwork
import tensorflow as tf

class NS2D_InformedNN_LOS(NeuralNetwork):
    def __init__(self, hp, logger, X_f, ub, lb):
        super().__init__(hp, logger, ub, lb)


        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.y_f = self.tensor(X_f[:, 1:2])
        self.t_f = self.tensor(X_f[:, 2:3])

    # Defining custom loss
    def loss(self, u, u_pred, x):
        f_c,f_u,f_v = self.f_model()

        los_pred = tf.math.multiply(u_pred[:, 0], u[:, 1])+tf.math.multiply(u_pred[:, 1], u[:, 2])
        los_true = u[:, 0]

        return tf.reduce_mean(tf.square(f_c))+tf.reduce_mean(tf.square(los_true-los_pred))\
               + tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_v))


    # The actual PINN
    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the three inputs we’ll need later, x ,y, and t
            tape.watch(self.x_f)
            tape.watch(self.y_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            X_f = tf.stack([self.x_f[:, 0], self.y_f[:, 0], self.t_f[:, 0]], axis=1)

            # Getting the prediction
            u, v = self.uv_model(X_f)

        # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
        u_x = tape.gradient(u, self.x_f)
        v_y = tape.gradient(v, self.y_f)
        u_t = tape.gradient(u, self.t_f)
        v_t = tape.gradient(v, self.t_f)
        u_y = tape.gradient(u, self.y_f)
        v_x = tape.gradient(v, self.x_f)

        f_u = u_t[:,0] + tf.math.multiply(u , u_x[:,0] ) + (tf.math.multiply(v, u_y[:,0] ))  #ignore diffussion and external force terms
        f_v = v_t[:,0]  + tf.math.multiply(u, v_x[:,0] )  + (tf.math.multiply(v, v_y[:,0] )) #ignore diffussion and external force terms

        # Letting the tape go
        del tape

        # Buidling the PINNs
        return u_x + v_y, f_u, f_v
            #u_t + u*u_x - nu*u_xx

    def get_params(self, numpy=False):
        return self.nu

    def predict(self, X_star):
        u_star = self.model(X_star)
        #f_star = self.f_model()
        return u_star.numpy()

    def uv_model(self, X):

        uv = self.model(X)
        u  = uv[:, 0]
        v  = uv[:, 1]

        return u, v





