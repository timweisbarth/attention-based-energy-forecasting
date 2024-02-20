import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        print("activation", self.activation)

    def forward(self, x, attn_mask=None):
        # x is (batch_size, sl, d_model)

        # Attention 
        new_x, attn = self.attention(
            x, x, x, 
            attn_mask=attn_mask
        ) # new_x is (batch_size, sl, d_model)

        # Add and norm
        x = x + self.dropout(new_x)
        y = x = self.norm1(x) # (batch_size, sl, d_model)

        # Feed forward, y.T is (batch_size, d_model, sl)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # (batch_size, d_ff, sl)
        y = self.dropout(self.conv2(y).transpose(-1, 1)) # (batch_size, sl, d_model)

        # Add and norm
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # self.conv_layers is None for Transformer
        print("Conv-layer", True if self.conv_layers is not None else False)
        attns = []
        if self.conv_layers is not None:
            print(len(self.attn_layers), len(self.conv_layers))
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                print("x_before attn_layer", x.shape)
                x, attn = attn_layer(x, attn_mask=attn_mask)
                print("x after attn_layer", x.shape)
                x = conv_layer(x)
                print("x after conv_layer", x.shape)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            print("x after for loop", x.shape)
            attns.append(attn)
        else:
            # self.attn_layers is a list of EncoderLayer
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # In our Transformer, we have a final normalization layer
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        # x is (batch_size, ll+pl, d_model), cross is (batch_size, sl, d_model)
        #print("x", x.shape)
        #print("cross", cross.shape)
        # Self attention (only need first element of tuple with is x)
        print("--------------- Decoder -------------------")
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0]) # x is (batch_size, ll+pl, d_model)
        
        x = self.norm1(x)
        print("Cross attention")
        # Cross attention
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        # Position-wise feed forward
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        print("--------------- Decoder End -------------------")
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        # Used for transformer
        if self.norm is not None:
            x = self.norm(x)

        # Also used for transformer
        if self.projection is not None:
            x = self.projection(x)
        return x
