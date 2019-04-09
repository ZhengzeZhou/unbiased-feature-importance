def UFI_cls(self, X, y):
    """Function for calculating unbiased measurement of feature importance 
    for RandomForestClassifier. Out-of-bag samples are used.

    Parameters  
    ----------
    self : RandomForestClassifier object. 
    X : array-like of shape = [n_samples, n_features]
        The training input samples. It should be the same data as you use 
        to fit RandomForestClassifier.
    y : array-like, shape = [n_samples]
        The target values (class labels in classification). Only binary 
        classsfication is supported currently.

    Returns
    -------
    feature importance: array, shape = [n_features]
    """   

    VI = np.array([0.] * self.n_features_)

    n_estimators = self.n_estimators

    for index, tree in enumerate(self.estimators_):

        temp = np.array([0.] * self.n_features_)

        n_nodes = tree.tree_.node_count

        tree_X_inbag = X.repeat((self.inbag_times_[:, index]).astype("int"), axis = 0)
        tree_y_inbag = y.repeat((self.inbag_times_[:, index]).astype("int"), axis = 0)
        decision_path_inbag = tree.decision_path(tree_X_inbag).todense()

        outofbag_index = (self.inbag_times_[:, index] == 0)
        tree_X = X[outofbag_index]
        tree_y = y[outofbag_index]
        decision_path = tree.decision_path(tree_X).todense()

        impurity = [0] * n_nodes

        weighted_n_node_samples = np.array(np.sum(decision_path, axis = 0))[0]

        for i in range(n_nodes):

            arr1 = tree_y[np.array(decision_path[:, i]).ravel().nonzero()[0].tolist()]
            arr2 = tree_y_inbag[np.array(decision_path_inbag[:, i]).ravel().nonzero()[0].tolist()]

            if len(arr1) == 0 or len(arr2) == 0:
                impurity[i] = 0
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

            incre = (weighted_n_node_samples[node] * impurity[node] -
                                weighted_n_node_samples[node_left] * impurity[node_left] -
                                weighted_n_node_samples[node_right] * impurity[node_right])

            temp[v] += incre

        VI += temp / len(tree_y)

    return VI / n_estimators
    