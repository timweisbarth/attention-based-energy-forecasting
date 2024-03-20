from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
import xgboost as xgb
from model_lstm import LSTMModel
#from optimization import Optimization
from model_transformer import Transformer
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def train(X_train, y_train, X_val, y_val, model_name, device):
    
    if model_name == "linreg":
        model = LinearRegression()
        model.fit(X_train, y_train)

        return model
    
    elif model_name == "xgb":

        # Convert your pandas DataFrame to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Specify the parameters for your XGBoost model, including 'gpu_hist' as the tree method to use the GPU
        params = {
            'tree_method': 'hist',  # Use GPU accelerated tree construction
            'device': device,
            'objective': 'reg:squarederror',
            'multi_strategy': "one_output_per_tree",
            'eval_metric': ['rmse'] 
        }

        # Train the model
        model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False)

        ############################ scikit-learn interface ##################        
        
        #model = xgb.XGBRegressor(
        #    n_estimators= 1000, 
        #    early_stopping_rounds=50, 
        #    multi_strategy="one_output_per_tree", 
        #    tree_method="hist", 
        #    device=device
        #)
        
        #model = xgb.XGBRegressor(
        #    tree_method="hist",
        #    n_estimators=128,
        #    n_jobs=16,
        #    max_depth=8,
        #    multi_strategy="multi_output_tree",
        #    subsample=0.6)
        
        #model.fit(X_train, y_train,
        #    eval_set=[(X_train, y_train), (X_val, y_val)],
        #    verbose=False,)
        
        #####################################################################
        
        return model
    
    elif model_name == "dummy":
        model = DummyRegressor(strategy='mean')
        model.fit(X_train, y_train)

        return model

class Optimization:
    def __init__(self, args, train_loader, val_loader):

        if args.model_name == "lstm":
            self.model = LSTMModel(args.model_params.input_dim, args.model_params.hidden_dim, args.model_params.layer_dim, args.model_params.output_dim, args.model_params.dropout_prob)
        elif args.model_name == "vanillaTransformer":
            self.model = Transformer(args.model_params.input_dim, args.model_params.n_embed, args.model_params.heads, args.model_params.n_blocks, args.model_params.output_dim)  

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            #else "mps"
            #if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.model = self.model.to(self.device)
        self.lr = args.train_params.learning_rate
        self.epochs = args.train_params.n_epochs
        self.wdecay = args.train_params.weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.plot_loss = args.plot_loss
        
        self.train_epoch_losses = []
        self.val_epoch_losses = []

        self.loss_fn = nn.MSELoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        

        print(f'Number of parameters in network = {sum(p.numel() for p in self.model.parameters())}\n')

    def train_step(self):
        self.model.train()
        num_train_batches = len(self.train_loader)
        train_batch_losses = 0
        for batch_number, (X, y) in enumerate(self.train_loader):
            yhat = self.model(X, y)
            batch_loss = self.loss_fn(yhat, y)
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_batch_losses += batch_loss.item()
        train_epoch_loss = train_batch_losses / num_train_batches
        self.train_epoch_losses.append(train_epoch_loss)
    
    def evaluate(self):
        self.model.eval()
        num_val_batches = len(self.val_loader)
        val_batch_losses = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                yhat = self.model(X)
                val_batch_losses += self.loss_fn(yhat, y).item()
        val_epoch_loss = val_batch_losses / num_val_batches
        self.val_epoch_losses.append(val_epoch_loss)
         
    def train(self):
        
        for epoch in range(1, self.epochs+1):
            
            self.train_step()
            self.evaluate()

            # Print results
            if ((epoch <= 10) | (epoch % max(int(self.epochs / 20), 1) == 0)) & self.plot_loss:
                        print(
                            f"[{epoch}/{self.epochs}] Training loss: {self.train_epoch_losses[-1]**0.5:.4f}\t Validation loss: {self.val_epoch_losses[-1]**0.5:.4f}"
                        )

        # Plot results
        if self.plot_loss:
            plt.plot(self.train_epoch_losses, label="Training loss")
            plt.plot(self.val_epoch_losses, label="Validation loss")
            plt.legend()
            plt.title("Losses")
            plt.show()

    def predict(self, data_loader):
        self.model.eval()
        num_batches = len(data_loader)
        batch_losses = 0
        preds = np.array([])
        vals = np.array([])
        with torch.no_grad():
            for X, y in data_loader:
                yhat = self.model(X)
                batch_losses += self.loss_fn(yhat, y).item()

                yhat = yhat.detach().numpy()
                y = y.detach().numpy()

                preds = np.vstack((preds, yhat)) if preds.size > 0 else yhat
                vals = np.vstack((vals, y)) if vals.size > 0 else y
        
        print("RMSE according to loss_fn:", (batch_losses/ num_batches)**0.5 )
        


        return preds, vals
    
