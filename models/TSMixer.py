import torch
from torch import nn
from torchtsmixer import TSMixer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        self.including_weather = configs.including_weather
        if self.including_weather:
            self.m = TSMixer(
                sequence_length=configs.seq_len + configs.pred_len,
                prediction_length=configs.pred_len,
                input_channels=configs.enc_in + 69,
                output_channels=configs.c_out,
                num_blocks=int(configs.e_layers),
                dropout_rate=configs.dropout,
                ff_dim=configs.d_model,
            )
        else:
            self.m = TSMixer(
                sequence_length=configs.seq_len,
                prediction_length=configs.pred_len,
                input_channels=configs.enc_in + 9,
                output_channels=configs.c_out,
                num_blocks=int(configs.e_layers),
                dropout_rate=configs.dropout,
                ff_dim=configs.d_model,
            )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        print("------------------")
        print(x_enc.shape, x_mark_enc.shape, x_dec.shape, x_mark_dec.shape)
        if self.including_weather:
            x_enc = torch.cat([x_enc, x_dec], dim=1)
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec], dim=1)
            x = torch.cat([x_enc, x_mark_enc], dim=-1)
            print(x_enc.shape, x_mark_enc.shape)
            print("x.shape", x.shape)
        else:
            x = torch.cat([x_enc, x_mark_enc], dim=-1)

        out = self.m(x)

        return out
