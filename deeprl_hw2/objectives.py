"""Loss functions."""

import tensorflow as tf
import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    diff = y_true - y_pred
    abs_diff = tf.abs(diff)
    sq_diff = tf.square(diff)
    # return diff**2/2. if diff <= max_grad else max_grad*diff - max_grad**2/2
    return tf.where(abs_diff <= max_grad, 0.5*sq_diff,
                    max_grad*abs_diff - 0.5*max_grad**2)


def mean_huber_loss(y_true, y_pred, max_grad=1., name='huber_loss'):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    # diff = abs(y_true - y_pred)
    # loss = diff**2/2. if diff <= max_grad else max_grad*diff - max_grad**2/2
    # return loss.mean()
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad), name=name)
