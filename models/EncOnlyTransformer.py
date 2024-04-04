import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(int(configs.e_layers))
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        #self.decoder = Decoder(
        #    [
        #        DecoderLayer(
        #            AttentionLayer(
        #                FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                configs.d_model, configs.n_heads),
        #            AttentionLayer(
        #                FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                configs.d_model, configs.n_heads),
        #            configs.d_model,
        #            configs.d_ff,
        #            dropout=configs.dropout,
        #            activation=configs.activation,
        #        )
        #        for l in range(configs.d_layers)
        #    ],
        #    norm_layer=torch.nn.LayerNorm(configs.d_model),
        #    projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        #)
        self.norm = nn.LayerNorm(configs.d_model)

        self.projection = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        self.projection2 = nn.Linear(configs.d_model, configs.c_out, bias=True)
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc (batch_size, sl, enc_in), x_mark_enc (batch_size, sl, time_f)
        # x_dec (batch_size, ll+pl, dec_in), x_mark_dec (batch_size, ll+pl, time_f)
        # masks = None, None, None
        
        
        out = self.enc_embedding(x_enc, x_mark_enc) # (batch_size, sl, d_model)
        out, attns = self.encoder(out, attn_mask=enc_self_mask) # (batch_size, sl, d_model)
        
        #dec_out = self.dec_embedding(x_dec, x_mark_dec) # (batch_size, ll+pl, d_model)
        #dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # (batch_size, ll+pl, c_out)
        
        
        # Used for transformer
        if self.norm is not None:
            out = self.norm(out)

        
        print(out.shape)
        # (batch_size, sl, d_model) -> (batch_size, d_model, pred_len) -> (batch_size, pred_len, d_model)
        out = self.projection(out.permute(0, 2, 1)).permute(0, 2, 1) 

        print(out.shape)
        out = self.projection2(out) # (batch_size, pred_len, c_out)
        print(out.shape)
        

        if self.output_attention:
            return out, attns
        else:
            return out  # (batch_size, pl, c_out)
