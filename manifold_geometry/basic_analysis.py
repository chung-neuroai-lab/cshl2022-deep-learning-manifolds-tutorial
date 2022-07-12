# Methods to calculate simple measures of geometric properties of manifolds
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def basic_analysis(X, n_iterations=10, svm=False):
    '''
    Computes the accuracy of a linear classifier trained to distinguish the objects in X, along
    with simple measures of the radius and dimension of the data for each object.

    Args:
        X: Sequence of arrays containing manifold data.  Each array is shape (N, P) where N is the
            feature dimension and P is the number of sampled points on the manifold.
        n_iterations: Number of different random train-test splits to evaluate classifier accuracy on
        svm: Optionally use a linear SVM.  Defaults to logistic regression otherwise.

    Returns:
        accuracies: Top-1 accuracies of a linear one vs. rest classifier trained to distinguish the
            different classes in X. Each element is computed on a different train/test split.
        radii: Simple measure of the radius of the activations for each object
        dimensions: Simple measure (participation ratio) of the dimension of the activations for
            for each object
    '''
    # Subtract the global mean
    Xori = np.concatenate(X, axis=1)
    global_mean = np.mean(Xori, axis=1, keepdims=True)
    X_centered = [x - global_mean for x in X]
    # Compute the radius of the data
    square_radii, _ = radius(X_centered)
    radii = np.sqrt(square_radii)
    # Compute participation ratio for each object
    participation_ratios = participation_ratio(X_centered)
    # Compute accuracy of a ovr linear SVM on the data
    accuracies = classification_performance(X_centered, split_ratio=0.8, num_iterations=n_iterations, svm=svm)
    return accuracies, radii, participation_ratios


def radius(X):
    '''
    Computes the radius of a sequence of manifolds.  Radius is defined by the average deviation
    from the manifold center.

    Args:
        X: Sequence of arrays containing manifold data.  Each array is shape (N, P) where N is the
            feature dimension and P is the number of sampled points on the manifold.

    Returns:
        square_radii: 1D array of length len(X) containing radius of the corresponding manifold
        r_stds: 1D array containing the standard deviation of the square radius for each manifold
    '''
    square_radii = []
    r_stds = []
    # Compute the radius for each manifold
    for data in X:
        center = np.mean(data, axis=1, keepdims=True) # (N, 1)
        center_norm = np.sum(np.square(center))
        centered_data = data - center # (N, P)
        square_deviation = np.diag(np.matmul(centered_data.T, centered_data))/center_norm # (P)
        square_radius = np.mean(square_deviation)
        square_radius_std = np.std(square_deviation)
        square_radii.append(square_radius)
    return square_radii, r_stds


def participation_ratio(X):
    '''
    Computes the participation ratio for a sequence of manifolds.  This is computed by
        PR = (sum_i mu_i)^2/(sum_i mu_i^2)
    Where mu_i is the i_th eigenvalue of the covariance matrix of the manifold.

    Args:
        X: Sequence of arrays containing manifold data.  Each array is shape (N, P) where N is the
            feature dimension and P is the number of sampled points on the manifold.

    Returns:
        participation_ratios: 1D array containing the participation ratio for each manifold
    '''
    participation_ratios = []
    # Compute the ratio for each manifold
    for data in X:
        covariance = np.matmul(data.T, data) # (N, N) covariance matrix
        mu, _ = np.linalg.eig(covariance) # (N) eigenvalues
        mu = np.real(mu)
        participation_ratio = np.square(np.sum(mu))/np.sum(np.square(mu))
        participation_ratios.append(participation_ratio)
    return participation_ratios


def classification_performance(X, split_ratio=0.8, num_iterations=1, train_accuracy=False, svm=False):
    '''
    Computes the accuracy of a linear classifier trained on the samples contained in X.
    A large number of samples may be neccesary for this to give an accurate result.

    Args:
        X: Sequence of arrays containing manifold data.  Each array is shape (N, P) where N is the
            feature dimension and P is the number of sampled points on the manifold.
        split_ratio: Fraction of data to include in training partition
        num_iterations: Number of splits to evaluate
        train_accuracy: If true report the accuracy on the training set
        svm: If true, use a linear SVM. Defaults to logistic regression otherwise.
    Returns:
        accuracies: Classification accuracy for a one vs. rest classifier trained on each manifold
    '''
    # Create the training data. sklearn expects a data array of shape (n_samples, n_features) and a label
    # array of shape (n_samples)
    X_trans = [x.T for x in X]
    X_data = np.concatenate(X_trans, axis=0)
    y_data = np.asarray([ i for i, x in enumerate(X_trans) for j in x ])
    data_idx = [i for i in range(X_data.shape[0])]
    # Shuffle the data to make train test splits
    accuracies = []
    for i in range(num_iterations):
        np.random.shuffle(data_idx)
        if split_ratio < 1:
            split_point = int(split_ratio * len(data_idx))
            assert split_point > 0
        else:
            split_point = len(data_idx)
        train_idx = data_idx[0:split_point]
        test_idx = data_idx[split_point:]
        # Split the training data and the labels
        X_train, X_test = X_data[train_idx, :], X_data[test_idx, :]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        # Normalize the data
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        X_train = (X_train - means)/stds
        X_test = (X_test - means)/stds
        # Train the classifier
        if svm:
            classifier = LinearSVC(class_weight='balanced', dual=False)
        else:
            classifier = LogisticRegression(class_weight='balanced', solver='liblinear', multi_class='ovr')
        classifier.fit(X_train, y_train)
        # Compute the mean accuracy
        if train_accuracy:
            accuracy = classifier.score(X_train, y_train)
        else:
            accuracy = classifier.score(X_test, y_test)
        accuracies.append(accuracy)
    return accuracies
