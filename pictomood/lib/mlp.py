"""
.. module:: mlp
    :synopsis: main module for multi-layer perceptron implementation

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 9, 2017
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.externals import joblib


class MLP:

    def __init__(self):
        pass

    def load_model(self, path=None):

        if path is None:
            self.model = MLPClassifier(
                solver='lbfgs',
                hidden_layer_sizes=(15,),
                random_state=1,
                max_iter=500
                # warm_start=True
            )
        else:
            self.model = joblib.load(path)

    def save_model(self, path):
        joblib.dump(self.model, path)

    def train(self, input_, output):

        kf = KFold(n_splits=10)

        for train_index, test_index in kf.split(input_):
            print('Fit:', self.model.fit(input_, output))
            # return self.model.score(input_, output)
            print('Score:', self.model.score(input_, output))

    def run(self, input_):
        return self.model.predict([input_])
