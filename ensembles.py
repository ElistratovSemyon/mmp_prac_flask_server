import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.fss = 1 if feature_subsample_size is None else feature_subsample_size
        self.kwargs = trees_parameters
        self.trees = []
        self.tree_features = []
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
        """
        val_f = False
        if not(X_val is None or y_val is None): 
            val = 0
            val_f = True
        for i in range(self.n_estimators):
            rows = np.random.randint(0, X.shape[0] , X.shape[0])
            cols = np.random.choice(np.arange(X.shape[1]), int(self.fss * X.shape[1]), replace=False)
            new_X, new_y = X[rows][:, cols], y[rows]
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            model.fit(new_X, new_y)
            self.trees.append(model)
            self.tree_features.append(cols)
            if val_f:
                val += model.predict(X_val[:, cols])
        if val_f:
            val /= self.n_estimators
            score = np.sqrt(((val - y_val) ** 2).mean())
            print("RMSE: %.4f" % score)
        return self
                      
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res = 0
        for i in range(self.n_estimators):
            cols = self.tree_features[i]
            res += self.trees[i].predict(X[:, cols])
        return res / self.n_estimators

class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        pass
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        pass

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pass

    def _RMSE(x, y):
        return  np.sqrt(((x - y)**2).mean())