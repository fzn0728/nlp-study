from tools import load_data, read_vocab, sigmoid, tanh, show_model
from nltk import WordNetLemmatizer, word_tokenize,download
#download('punkt')
import numpy as np

"""
This cell shows you how the model will be used, you have to finish the cell below before you
can run this cell. 

Once the implementation is done, you should hype tune the parameters to find the best config
"""

from sklearn.model_selection import train_test_split
data = load_data("train.txt")
vocab = read_vocab("vocab.txt")
X, y = data.text, data.target
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3)

class CNNTextClassificationModel:
    def __init__(self, vocab, window_size=2, F=100, alpha=0.1):
        """
        F: number of filters
        alpha: back propagatoin learning rate
        """
        self.vocab = vocab
        self.window_size = window_size
        self.F = F
        self.alpha = alpha

        # U and w are the weights of the hidden layer, see Fig 1 in the pdf file
        # U is the 1D convolutional layer with shape: voc_size * num_filter * window_size
        self.U = np.random.normal(loc=0, scale=0.01, size=(len(vocab), F, window_size))
        # w is the weights of the activation layer (after max pooling)
        self.w = np.random.normal(loc=0, scale=0.01, size=(F + 1))

    def pipeline(self, X):
        """
        Data processing pipeline to:
        1. Tokenize, Normalize the raw input
        2. Translate raw data input into numerical encoded vectors

        :param X: raw data input
        :return: list of lists

        For example:
        X = ["Apples orange banana",
         "orange apple bananas"]
        returns:
        [[0, 1, 2],
         [0, 2, 3]]
        """
        X2 = []
        unknown = vocab['__unknown__']
        default = vocab['.']
        wnet = WordNetLemmatizer()

        for i in range(len(X)):
            cleaned_tokens = [self.vocab.get(wnet.lemmatize(w), unknown) for w in word_tokenize(X[i])]
            if len(cleaned_tokens) < self.window_size:
                cleaned_tokens = cleaned_tokens + [default] * (self.window_size - len(cleaned_tokens))
            X2.append(cleaned_tokens)

        return X2

    @staticmethod
    def accuracy(probs, labels):
        assert len(probs) == len(labels), "Wrong input!!"
        a = np.array(probs)
        b = np.array(labels)

        return 1.0 * (a == b).sum() / len(b)

    def train(self, X_train, y_train, X_dev, y_dev, nEpoch=50):
        """
        Function to fit the model
        :param X_train, X_dev: raw data input
        :param y_train, y_dev: label
        :nEpoch: number of training epoches
        """
        X_train = self.pipeline(X_train)
        X_dev = self.pipeline(X_dev)

        for epoch in range(nEpoch):
            self.fit(X_train, y_train)

            accuracy_train = self.accuracy(self.predict(X_train), y_train)
            accuracy_dev = self.accuracy(self.predict(X_dev), y_dev)

            print("Epoch: {}\tTrain accuracy: {:.3f}\tDev accuracy: {:.3f}"
                  .format(epoch, accuracy_train, accuracy_dev))

    def fit(self, X, y):
        """
        :param X: numerical encoded input
        """
        for (data, label) in zip(X, y):
            self.backward(data, label)

        return self

    def predict(self, X):
        """
        :param X: numerical encoded input
        """
        result = []
        for data in X:
            if self.forward(data)["prob"] > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result

    def forward(self, word_indices):
        """
        :param word_indices: a list of numerically ecoded words
        :return: a result dictionary containing 3 items -
        result['prob']: \hat y in Fig 1.
        result['h']: the hidden layer output after max pooling, h = [h1, ..., hf]
        result['hid']: argmax of F filters, e.g. j of x_j
        e.g. for the ith filter u_i, tanh(word[hid[j], hid[j] + width]*u_i) = h_i
        """
        assert len(word_indices) >= self.window_size, "Input length cannot be shorter than the window size"

        h = np.zeros(self.F + 1, dtype=float)
        hid = np.zeros(self.F, dtype=int)
        prob = 0.0

        # layer 1. compute h and hid
        # loop through the input data of word indices and
        # keep track of the max filtered value h_i and its position index x_j
        # h_i = max(tanh(weighted sum of all words in a given window)) over all windows for u_i
        """
        Implement your code here
        """
        for F_index in range(len(self.U[0])):
            temp_list = []
            for word_indices_ind in range(len(word_indices) - self.window_size + 1):
                window_sum = 0.0
                for window_ind in range(self.window_size):
                    window_sum += self.U[word_indices[word_indices_ind + window_ind]][F_index][window_ind]
                #print(word_indices_ind, self.U[word_indices[word_indices_ind+window_ind]][F_index][window_ind])
                temp_list.append(tanh(window_sum))
            h[F_index] = np.max(temp_list)
            hid[F_index] = np.argmax(temp_list)
        h[-1] = 1

        # layer 2. compute probability
        # once h and hid are computed, compute the probabiliy by sigmoid(h^TV)
        """
        Implement your code here
        """
        prob_sum = 0.0
        for w_i, h_i in zip(self.w, h):
            prob_sum += w_i * h_i

        prob = sigmoid(prob_sum)
        # return result
        return {"prob": prob, "h": h, "hid": hid}

    def backward(self, word_indices, label):
        """
        Update the U, w using backward propagation

        :param word_indices: a list of numerically ecoded words
        :param label: int 0 or 1
        :return: None

        update weight matrix/vector U and V based on the loss function
        """

        pred = self.forward(word_indices)
        prob = pred["prob"]
        h = pred["h"]
        hid = pred["hid"]

        L = -(label) * np.log(prob) - (1 - label) * np.log(1 - prob)
        #         print(L)
        # update U and w here
        # to update V: w_new = w_current + d(loss_function)/d(w)*alpha
        # to update U: U_new = U_current + d(loss_function)/d(U)*alpha
        # Hint: use Q6 in the first part of your homework
        """
        Implement your code here
        """
        old_w = self.w[:]
        self.w = self.w + (label - prob) * h * self.alpha
        #         print(h)
        for F_index in range(self.F):
            incre = (label - prob) * old_w[F_index] * (1 - h[F_index] ** 2)
            #             print((label - prob),old_w[F_index],(1-h[F_index]**2))
            for i in range(self.window_size):
                self.U[word_indices[hid[F_index] + i]][F_index][i] += incre * self.alpha


cls = CNNTextClassificationModel(vocab)
cls.train(X_train, y_train, X_dev, y_dev, nEpoch=10)