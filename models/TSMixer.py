import torch
from torch import nn
from torchtsmixer import TSMixerExt, TSMixer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len
        #self.m = TSMixerExt(
        #    sequence_length=configs.seq_len,
        #    prediction_length=configs.pred_len,
        #    input_channels=configs.enc_in,
        #    extra_channels=9,
        #    hidden_channels=8,
        #    static_channels=1,
        #    output_channels=configs.c_out
        #)
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
        x_extra_future = x_mark_dec[:,-self.pred_len:,:]
        x = torch.cat([x_enc, x_mark_enc], dim=-1)
        out = self.m(x)
        #out = self.m(x_hist=x_enc,
        #            x_extra_hist=x_mark_enc,
        #            x_extra_future=x_extra_future,
        #            x_static=torch.ones(32,1) / self.pred_len,
        #            )
        return out
