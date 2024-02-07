from layers.Embed import DataEmbedding
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()


        self.e_layers = configs.e_layers
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        # Embedding of data
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # LSTM layers
        self.lstm = nn.LSTM(
            configs.d_model, configs.d_model, configs.e_layers, batch_first=True, dropout=configs.dropout
        )

        # Fully connected layer
        self.fc = nn.Linear(configs.d_model, configs.pred_len * configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        #print("x_enc", x_enc.shape)
        #print("x_mark", x_mark_enc.shape)
        # Embed input 
        x = self.enc_embedding(x_enc, x_mark_enc) # (B, T, d_model)
        #print("x", x.shape)
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.e_layers, x.size(0), self.d_model).requires_grad_() # (e_layers, B, d_model)
        #print("h0", h0.shape)
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.e_layers, x.size(0), self.d_model).requires_grad_() # (e_layers, B, d_model)

        # Detaching is not strictly necessary but doesn't hurt aswell
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        #print(out.shape)
        out = out[:, -1, :]
        #print(out.shape)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        #print(out.shape)
        out = out.view(out.shape[0], self.pred_len, self.c_out)
        #print("out", out.shape)
        return out