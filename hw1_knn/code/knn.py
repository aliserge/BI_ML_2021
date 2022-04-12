import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print("Train X recorded")

    def foo(self):
        print("foo")

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.y_train)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X_test_01):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        """
        YOUR CODE IS HERE
        """
        print("Entering distance computing - 2Loops")
        answer = np.zeros((len(X_test_01), len(self.X_train)))
        
        for i in range(len(X_test_01)):
            for j in range(len(self.X_train)):
                answer[i,j] = self.L1(X_test_01[i], self.X_train[j])
                #print('in')
        print(answer)
        return answer
        
    def L1(self, a, b):
        return sum(np.abs(a - b))


    def compute_distances_one_loop(self, X_test_01):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        self.answer = np.zeros((len(X_test_01), len(self.X_train)))
        
        for i in range(len(X_test_01)):
            test = X_test_01[i]
            La = lambda b: self.L1(test, b)
            self.answer[i] = np.apply_along_axis(La, 1, self.X_train)
        return self.answer
        


    def compute_distances_no_loops(self, X_test_01):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        return np.sum(np.abs(self.X_train[None,:,:]-X_test_01[:,None,:]),axis=2)
        

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        #prediction = np.zeros(n_test)
        
        """
        YOUR CODE IS HERE
        """
        
        pre_prediction = self.y_train[np.argpartition(distances, self.k, axis = 1)[:,:self.k]]
        groups = np.unique(pre_prediction)
        counts = np.array([np.count_nonzero(pre_prediction == group, axis=1) for group in groups]).T

        indexes = np.argmax(counts, axis=1)
        return groups[indexes]
        
    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        """
        YOUR CODE IS HERE
        """
        pre_prediction = self.y_train[np.argpartition(distances, self.k, axis = 1)[:,:self.k]]
        groups = np.unique(pre_prediction)
        counts = np.array([np.count_nonzero(pre_prediction == group, axis=1) for group in groups]).T

        indexes = np.argmax(counts, axis=1)
        return groups[indexes]
