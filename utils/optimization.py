from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class LastValueNaiveRegressor(BaseEstimator, RegressorMixin):
    """Naive regressor that predicts the last value of each input sequence."""
    
    def fit(self, X, y=None):
        """Doesn't need to do anything"""
        self.horizon = y.shape[1]
        return self
    
    def predict(self, X):
        """Return predictions"""
        # Take the last value from each sequence in X and repeat it n_steps times
        out = np.repeat(X["wind_gen"].values.reshape(-1,1), self.horizon, axis=1)
        return out


def train(X_train, y_train, X_val, y_val, model_name, device, train_params):
    """
    Train model according to model_name on device 
    with train_params and return the trained model.
    """
    
    if model_name == "linreg":
        model = LinearRegression()
        model.fit(X_train, y_train)

        return model
    
    elif model_name == "ridge":
        model= Ridge(alpha=10)
        model.fit(X_train, y_train)
        print("Training Ridge")

        return model
    
    elif model_name == "xgb":

        # Convert your pandas DataFrame to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Specify the parameters for your XGBoost model, including 'gpu_hist' 
        # as the tree method to use the GPU
        params = {
            'tree_method': 'hist',  # Use GPU accelerated tree construction
            'device': device,
            'objective': 'reg:squarederror',
            'multi_strategy': "one_output_per_tree",
            'eval_metric': ['rmse'] 
        }


        add_params = [{},
                      {"learning_rate": 0.15}, {"learning_rate": 0.45}, 
                      {"max_depth": 3}, {"max_depth": 9},
                      {"subsample": 0.5}, {"subsample": 0.75}, 
                      {"colsample_bytree": 0.5}, {"colsample_bytree": 0.75}, 
                      {"lambda": 0.1}, {"lambda": 10}, 
                      {"alpha": 0.1}, {"alpha": 1}]
        
        params.update(add_params[train_params.run])

        # Train the model
        # num_boost_round is the number of boosting rounds or trees to build, 
        # early_stopping determines how many will actually be built
        model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')], verbose_eval=True, num_boost_round = 1000, early_stopping_rounds = 50)
        
        return model
    
    elif model_name == "dummy":
        model = DummyRegressor(strategy='mean')
        model.fit(X_train, y_train)

        return model
    elif model_name == "repeat":
        model = LastValueNaiveRegressor()
        model.fit(X_train, y_train)
        return model
