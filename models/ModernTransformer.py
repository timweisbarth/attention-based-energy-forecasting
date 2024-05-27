import torch
import torch.nn as nn
from layers.ModernTransformer_EncDec import PreLNDecoder, DecoderLayer, PreLNEncoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
import math
import pandas as pd

#def create_patched_version(x, P, S, pad_len, new_sl):
#    bs, sl, ch = x.shape

    # Pad the tensor by repeating the last entry
#    x_padded = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
#    
#    # Extract patches
#    patches = x_padded.unfold(dimension=1, size=P, step=S)
#    
#    # Reshape patches to the desired shape (bs, new_sl, P*ch)
#    patches = patches.reshape(bs, new_sl, P * ch)
#    
#    return patches


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.enc_pad_len = self.stride - (self.seq_len - self.patch_len) % self.stride
        #self.enc_new_sl = (self.seq_len - self.patch_len + self.enc_pad_len) // self.stride + 1

        self.dec_pad_len = self.stride - (self.label_len + self.pred_len - self.patch_len) % self.stride
        #self.dec_new_sl = (self.label_len + self.pred_len - self.patch_len + self.dec_pad_len) // self.stride + 1

        # Embedding
        self.enc_embedding = ModernDataEmbedding(configs.d_model, self.patch_len, self.stride, self.enc_pad_len)
        self.dec_embedding = ModernDataEmbedding(configs.d_model, self.patch_len, self.stride, self.dec_pad_len)
        self.cross_att_embedding = ...

        # Encoder
        self.encoder = PreLNEncoder(
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
        self.decoder = PreLNDecoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc (batch_size, sl, enc_in), x_mark_enc (batch_size, sl, time_f)
        # x_dec (batch_size, ll+pl, dec_in), x_mark_dec (batch_size, ll+pl, time_f)
        # masks = None, None, None


        # Version 1 which first creates the patches and then would apply nn.Linear
        #x_enc = create_patched_version(x_enc, P=self.patch_len, S=self.stride, pad_len=self.enc_pad_len, new_sl=self.enc_new_sl)
        #x_mark_enc = create_patched_version(x_mark_enc, P=self.patch_len, S=self.stride, pad_len=self.enc_pad_len, new_sl=self.enc_new_sl)

        #x_dec = create_patched_version(x_dec, P=self.patch_len, S=self.stride, pad_len=self.dec_pad_len, new_sl=self.dec_new_sl)
        #x_mark_dec = create_patched_version(x_mark_dec, P=self.patch_len, S=self.stride, pad_len=self.dec_pad_len, new_sl=self.dec_new_sl)

        # Cross Attention embedding: Watch out for masking, 
        #self.dropout(self.cross_attention(
        #    x, cross, cross,
        #    attn_mask=cross_mask
        #)[0])

        x_enc = pd.concat([x_enc, x_mark_enc], axis=2)
        enc_out = self.enc_embedding(x_enc) # (batch_size, enc_new_sl, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) # (batch_size, enc_new_sl, d_model)
        

        # TODO!: Decoder dec_new_sl != pred_len due to patching
        # TODO Masking, Cross Attention, individual embeddings for each similar data (Verify via not normalized data), bias or no for embedding? (Add time first such that not only zeros)
        # TODO: Which transformer architecture exactely?
        x_dec = pd.concat([x_dec, x_mark_dec], axis=2)
        dec_out = self.dec_embedding(x_dec) # (batch_size, dec_new_sl, d_model)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # (batch_size, ll+pl, c_out)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # (batch_size, pl, c_out)
        
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(1, max_len, d_model)
        self.positional_embedding = nn.Parameter(pe)
    def forward(self, x):
        return self.positional_embedding[:, :x.size(1), :]


class ModernDataEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, pad_len):
        super(ModernDataEmbedding, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=84, out_channels=d_model, kernel_size=patch_len, stride=stride, padding=(0, pad_len), bias=False, padding_mode='replicate')
        self.positional_embedding = PositionalEmbedding(d_model=d_model, max_len=1000)
    
    def forward(self, x):
        x = self.conv1d(x.permute(0,2,1)).permute(0,2,1)
        return x + self.positional_embedding(x)
