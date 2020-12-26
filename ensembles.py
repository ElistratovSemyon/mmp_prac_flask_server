import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(self, n_estimators=100, max_depth=None, feature_subsample_size=None,
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
        
    def fit(self, X, y, X_val=None, y_val=None, is_fitted=False):
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
        if not is_fitted:
            self.trees = []
            self.tree_features = []
        val_f = False
        if not(X_val is None or y_val is None): 
            val = 0
            val_f = True
            val_score_verbose = []
        for i in range(1, self.n_estimators+1):
            rows = np.random.randint(0, X.shape[0] , X.shape[0])
            cols = np.random.choice(np.arange(X.shape[1]), int(self.fss * X.shape[1]), replace=False)
            new_X, new_y = X[rows][:, cols], y[rows]
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            model.fit(new_X, new_y)
            self.trees.append(model)
            self.tree_features.append(cols)
            if val_f:
                val += model.predict(X_val[:, cols])
                val_score_verbose.append(np.sqrt(((val/i - y_val) ** 2).mean()))
        if val_f:
            val /= self.n_estimators
            score = np.sqrt(((val - y_val) ** 2).mean())
            print("RMSE: %.4f" % score)
            return val_score_verbose
        else:
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
        for i in range(1, self.n_estimators+1):
            cols = self.tree_features[i-1]
            res += self.trees[i-1].predict(X[:, cols])
        return res / self.n_estimators
    
    def get_params(self):
        tree_params = {}
        tree_params["n_estimators"] = self.n_estimators
        tree_params["feature_subsample_size"] = self.fss
        if len(self.trees) == 0:
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            tree_params.update(model.get_params())
        else:
            tree_params.update(self.trees[0].get_params())
        tree_params.pop("max_features")
        return tree_params

class GradientBoostingMSE:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
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
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.fss = 1 if feature_subsample_size is None else feature_subsample_size
        self.kwargs = trees_parameters
        self.trees = []
        self.tree_features = []
        self.loss = self._RMSE
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        w = np.ones(self.n_estimators)
        f = np.zeros_like(y)
        g = np.empty(self.n_estimators)
        val_f = False
        if not(X_val is None or y_val is None): 
            val = 0
            val_f = True
            val_score_verbose = []
        for i in range(self.n_estimators):
            grad = (y - f)
            cols = np.random.choice(np.arange(X.shape[1]), int(self.fss * X.shape[1]), replace=False)            
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            model.fit(X[:, cols], grad)
            res = model.predict(X[:, cols])
            g[i] = (grad * res).sum() / (res ** 2).sum() 
            f += self.learning_rate * g[i] * res
            self.trees.append(model)
            self.tree_features.append(cols)
            if val_f:
                val += model.predict(X_val[:, cols]) * g[i]
                val_score_verbose.append(np.sqrt(((val * self.learning_rate - y_val) ** 2).mean()))
        self.g = g
        if val_f:
            return val_score_verbose
        else:
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
            res += self.trees[i].predict(X[:, self.tree_features[i]]) * self.g[i]
        return res * self.learning_rate

    def get_params(self):
        tree_params = {}
        tree_params["n_estimators"] = self.n_estimators
        tree_params["learning_rate"] = self.learning_rate
        tree_params["feature_subsample_size"] = self.fss
        if len(self.trees) == 0:
            model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
            tree_params.update(model.get_params())
        else:
            tree_params.update(self.trees[0].get_params())
        tree_params.pop("max_features")
        return tree_params

    def _RMSE(x, y):
        return  np.sqrt(((x - y)**2).mean())