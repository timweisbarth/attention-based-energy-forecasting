from layers.Embed import DataEmbedding
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()


        self.e_layers = int(configs.e_layers)
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.device = device
        self.including_weather = configs.including_weather

        # Embedding of data
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, self.including_weather)

        # LSTM layers
        self.lstm = nn.LSTM(
            configs.d_model, configs.d_model, self.e_layers, batch_first=True, dropout=configs.dropout
        )
     

        if self.including_weather:
            self.dec_embedding = nn.Linear(69, configs.d_model)
            self.dec_lstm = nn.LSTM(
                configs.d_model, configs.d_model, self.e_layers, batch_first=True, dropout=configs.dropout
            )
            self.fc = nn.Linear(configs.d_model, configs.c_out)
        else:
            # Fully connected layer
            self.fc = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # Embed input 
        x = self.enc_embedding(x_enc, x_mark_enc) # (B, T, d_model)
        
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.e_layers, x.size(0), self.d_model, device=self.device).requires_grad_() # (e_layers, B, d_model)
        
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.e_layers, x.size(0), self.d_model, device=self.device).requires_grad_() # (e_layers, B, d_model)

        # Detaching is not strictly necessary but doesn't hurt aswell
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        if self.including_weather:
            x_mark_dec = x_mark_dec[:,-self.pred_len:,:]
            x_dec = self.dec_embedding(x_mark_dec)
            out, (hn, cn) = self.dec_lstm(x_dec, (hn, cn))
            out = self.fc(out)

        else:
            out = out[:, -1, :]
            # Convert the final state to our desired output shape (batch_size, output_dim)
            out = self.fc(out)
            out = out.view(out.shape[0], self.pred_len, self.c_out)
        
        return out