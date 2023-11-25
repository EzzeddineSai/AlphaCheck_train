import tensorflow as tf

def alpha_zero_loss(y_true, y_pred):
        squared_difference = (y_true[:,256] - y_pred[:,256])**2
        cce = tf.keras.losses.CategoricalCrossentropy()
        cce_loss = cce(y_true[:,:256],y_pred[:,:256])
        return squared_difference+cce_loss