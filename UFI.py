import numpy as np
from sklearn.utils import check_random_state

def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices



def inbag_times_(self, X):
    """ Return n_samples by n_estimators array which keeps track of which samples are
        "in-bag" in which trees for how many times.

    Parameters  
    ----------
    self : RandomForest object. 
    X : array-like of shape = [n_samples, n_features]
    The training input samples. It should be the same data as you use to fit RandomForest. 

    Returns
    -------
    inbag_times_ : array, shape = [n_samples, n_estimators]
    """
    n_samples = X.shape[0]
    inbag = np.zeros((n_samples, self.n_estimators))
    for t_idx in range(self.n_estimators):
        sample_idx = _generate_sample_indices(self.estimators_[t_idx].random_state, n_samples)
        inbag[:, t_idx] = np.bincount(sample_idx, minlength = n_samples)

    return inbag


def cls(self, X, y):
    """Return unbiased measurement of feature importance or RandomForestClassifier using out-of-bag samples.

    Parameters  
    ----------
    self : RandomForestClassifier object. 
    X : array-like of shape = [n_samples, n_features]
        The training input samples. It should be the same data as you use to fit RandomForestClassifier.
    y : array-like of shape = [n_samples]
        The target values (class labels in classification). Only binary classsfication is supported currently.

    Returns
    -------
    feature importance: array, shape = [n_features]
    """   
    VI = np.array([0.] * self.n_features_)

    n_estimators = self.n_estimators

    inbag = inbag_times_(self, X)

    for index, tree in enumerate(self.estimators_):

        temp = np.array([0.] * self.n_features_)

        n_nodes = tree.tree_.node_count

        tree_X_inb = X.repeat((inbag[:, index]).astype("int"), axis = 0)
        tree_y_inb = y.repeat((inbag[:, index]).astype("int"), axis = 0)
        decision_path_inb = tree.decision_path(tree_X_inb).todense()

        tree_X_oob = X[inbag[:, index] == 0]
        tree_y_oob = y[inbag[:, index] == 0]
        decision_path_oob = tree.decision_path(tree_X_oob).todense()

        impurity = [0] * n_nodes

        flag = [True] * n_nodes

        weighted_n_node_samples = np.array(np.sum(decision_path_inb, axis = 0))[0] / tree_X_inb.shape[0]

        for i in range(n_nodes):

            arr1 = tree_y_oob[np.array(decision_path_oob[:, i]).ravel().nonzero()[0].tolist()]
            arr2 = tree_y_inb[np.array(decision_path_inb[:, i]).ravel().nonzero()[0].tolist()]

            if len(arr1) == 0:
                if sum(tree.tree_.children_left == i) > 0:
                    parent_node = np.arange(n_nodes)[tree.tree_.children_left == i][0]
                    flag[parent_node] = False
                else:
                    parent_node = np.arange(n_nodes)[tree.tree_.children_right == i][0]
                    flag[parent_node] = False
            else:
                p1 = float(sum(arr1)) / len(arr1)
                pp1 = float(sum(arr2)) / len(arr2)
                p2 = 1 - p1
                pp2 = 1- pp1

                impurity[i] = 1 - p1 * pp1 - p2 * pp2

        for node in range(n_nodes):

            if tree.tree_.children_left[node] == -1 or tree.tree_.children_right[node] == -1:
                continue

            v = tree.tree_.feature[node]

            node_left = tree.tree_.children_left[node]
            node_right = tree.tree_.children_right[node]

            if flag[node] == True:

                incre = (weighted_n_node_samples[node] * impurity[node] -
                         weighted_n_node_samples[node_left] * impurity[node_left] -
                         weighted_n_node_samples[node_right] * impurity[node_right])

                temp[v] += incre

        VI += temp

    return VI / n_estimators


def regr(self, X, y):
    """Return unbiased measurement of feature importance for RandomForestRegressor using out-of-bag samples.

    Parameters  
    ----------
    self : RandomForestRegressor object. 
    X : array-like of shape = [n_samples, n_features]
        The training input samples. It should be the same data as you use to fit RandomForestRegressor.
    y : array-like of shape = [n_samples]
        The target values (real numbers in regression). 

    Returns
    -------
    feature importance: array, shape = [n_features]
    """   
    VI = np.array([0.] * self.n_features_)

    n_estimators = self.n_estimators

    inbag = inbag_times_(self, X)

    for index, tree in enumerate(self.estimators_):

        temp = np.array([0.] * self.n_features_)

        n_nodes = tree.tree_.node_count
        
        tree_X_inb = X.repeat((inbag[:, index]).astype("int"), axis = 0)
        tree_y_inb = y.repeat((inbag[:, index]).astype("int"), axis = 0)
        decision_path_inb = tree.decision_path(tree_X_inb).todense()

        tree_X_oob = X[inbag[:, index] == 0]
        tree_y_oob = y[inbag[:, index] == 0]
        decision_path_oob = tree.decision_path(tree_X_oob).todense()

        impurity_train = tree.tree_.impurity
        impurity = [0] * n_nodes

        flag = [True] * n_nodes

        weighted_n_node_samples = np.array(np.sum(decision_path_inb, axis = 0))[0] / tree_X_inb.shape[0]

        for i in range(n_nodes):

            arr1 = tree_y_oob[np.array(decision_path_oob[:, i]).ravel().nonzero()[0].tolist()]
            arr2 = tree_y_inb[np.array(decision_path_inb[:, i]).ravel().nonzero()[0].tolist()]

            if len(arr1) == 0:
                if sum(tree.tree_.children_left == i) > 0:
                    parent_node = np.arange(n_nodes)[tree.tree_.children_left == i][0]
                    flag[parent_node] = False
                else:
                    parent_node = np.arange(n_nodes)[tree.tree_.children_right == i][0]
                    flag[parent_node] = False
            else:
                impurity[i] = np.sum((arr1 - np.mean(arr2)) ** 2) / len(arr1)

        for node in range(n_nodes):

            if tree.tree_.children_left[node] == -1 or tree.tree_.children_right[node] == -1:
                continue

            v = tree.tree_.feature[node]

            node_left = tree.tree_.children_left[node]
            node_right = tree.tree_.children_right[node]

            if flag[node] == True:

                incre = (weighted_n_node_samples[node] * (impurity[node] + impurity_train[node]) -
                         weighted_n_node_samples[node_left] * (impurity[node_left] + impurity_train[node_left]) - 
                         weighted_n_node_samples[node_right] * (impurity_train[node_right] + impurity[node_right]))

                temp[v] += incre

        VI += temp 

    return VI / n_estimators