"""

Various model interfaces.

"""


class Model(object):
    """Base model class."""
    def predict(self, X):
        """Predict score for X.

        Parameters
        ----------
        X : array_like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted scores.

        """
        raise NotImplementedError()


class AdditiveModel(Model):
    """Additive models enable useful tools such as early stoppage."""
    def iter_y_delta(self, i, X):
        """Calculates target deltas for one iteration of the model.

        Parameters
        ----------
        i : iteration for which to get deltas
        X : array_like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y_delta : array of shape = [n_samples]
            y_delta[j] = ensemble[:i + 1](X[j]) - ensemble[:i](X[j])

        """
        raise NotImplementedError()

    def trim(self, n):
        """Trim model to first n iterations.

        Parameters
        ----------
        n : number of iterations to keep

        """
        raise NotImplementedError()
