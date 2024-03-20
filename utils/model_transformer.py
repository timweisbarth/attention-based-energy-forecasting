import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    """One head of standard self attention"""
    def __init__(self, n_embed, head_size):
        super().__init__()

        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
    def forward(self, x):
        # x is (B, T, n_embed)
        q, k, v = self.query(x), self.key(x), self.value(x) # (B,T,head_size)
        B,T,head_size = q.shape
        affinities = q @ k.transpose(-2,-1) * head_size**(-0.5)  # (B,T,head_size) @ (B,head_size,T) --> (B,T,T)
        att = F.softmax(affinities, dim=-1) @ v # (B, T, head_size)
        return att
    
class MaskedAttentionHead(nn.Module):
    """One head of masked self attention"""
    def __init__(self, n_embed, head_size):
        super().__init__()

        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(48, 48)))
        
    def forward(self, x):
        # x is (B, T, n_embed)
        q, k, v = self.query(x), self.key(x), self.value(x) # (B,T_dec,head_size)
        B,T,head_size = q.shape
        affinities = q @ k.transpose(-2,-1) * head_size**(-0.5)  # (B,T_dec,head_size) @ (B,head_size,T_dec) --> (B,T_dec,T_dec)
        affinities = affinities.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        att = F.softmax(affinities, dim=-1) @ v # (B, T_dec, T_dec) @ (B,T_dec,head_size) (B, T_dec, head_size)
        return att
    

class CrossAttentionHead(nn.Module):
    """One head of cross attention"""
    def __init__(self, n_embed, head_size):
        super().__init__()

        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        
    def forward(self, x, x_enc):
        # x is (B, T, n_embed)
        k, v = self.key(x_enc), self.value(x_enc) # (B,T_enc,head_size)
        q = self.query(x) # (B, T_dec, head_size)
        B,T,head_size = q.shape
        affinities = q @ k.transpose(-2,-1) * head_size**(-0.5)  # (B,T_dec,head_size) @ (B,head_size,T_enc) --> (B,T_dec,T_enc)
        att = F.softmax(affinities, dim=-1) @ v # (B, T_dec, T_enc) @ (B,T_enc,head_size) (B, T_dec, head_size)
        return att
    
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention of type attention_type"""
    def __init__(self, n_embed, heads, attention_type):
        super().__init__()

        head_size = n_embed // heads
        self.attention_type = attention_type
        if self.attention_type == "normal":
            self.multihead = nn.ModuleList([AttentionHead(n_embed, head_size) for _ in range(heads)])
        elif self.attention_type == "masked":
            self.multihead = nn.ModuleList([MaskedAttentionHead(n_embed, head_size) for _ in range(heads)])
        elif self.attention_type == "cross":
            self.multihead = nn.ModuleList([CrossAttentionHead(n_embed, head_size) for _ in range(heads)])
        self.linear = nn.Linear(n_embed, n_embed)
    
    def forward(self, x, x_enc=None):
        if self.attention_type == "cross":
            out = torch.cat([h(x, x_enc) for h in self.multihead], dim=-1)
        else:
            out= torch.cat([h(x) for h in self.multihead], dim=-1)
        
        out = self.linear(out)

        return out
    
class MLP(nn.Module):
    """A simple feed forward mlp with ReLU non-linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed)
        )
        
    def forward(self, x):
        return self.net(x)
    
class EncoderBlock(nn.Module):
    """Transformer encoder block: Multi-Head Attention followd by MLP"""
    def __init__(self, n_embed, heads):
        super().__init__()

        self.attention = MultiHeadAttention(n_embed, heads, "normal")
        self.mlp = MLP(n_embed)

        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = self.layer_norm1(self.attention(x) + x)
        x = self.layer_norm2(self.mlp(x) + x)
        return x

class DecoderBlock(nn.Module):
    """Transformer decoder block: 2xMulti-Head Attention followed by MLP"""
    def __init__(self, n_embed, heads):
        super().__init__()
        
        self.masked_attention = MultiHeadAttention(n_embed, heads, "masked")
        self.cross_attention = MultiHeadAttention(n_embed, heads, "cross")
        self.mlp = MLP(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)
        self.layer_norm3 = nn.LayerNorm(n_embed)
        
    def forward(self, x, x_enc):
        x = self.layer_norm1(self.masked_attention(x) + x) 
        x = self.layer_norm2(self.cross_attention(x, x_enc) + x)
        x = self.layer_norm3(self.mlp(x) + x)
        return x

class Transformer(nn.Module):
    """Transformer consisting of n_blocks of encoders and decoders"""
    def __init__(self, n_input, n_embed, heads, n_blocks, horizon):
        super().__init__()

        self.embed = nn.Linear(n_input, n_embed)
        self.embed2 = nn.Linear(1, n_embed)
        self.encoder = nn.Sequential(*[EncoderBlock(n_embed, heads) for _ in range(n_blocks)])
        self.decoder = nn.Sequential(*[DecoderBlock(n_embed, heads) for _ in range(n_blocks)])
        self.linear = nn.Linear(n_embed, 1)
        self.horizon = horizon

    def forward(self, x, y_true=None):

        # Encoder
        x_embed = self.embed(x) # (B,T,n_embed)
        B,T,n_embed = x_embed.shape
        
        x_enc = self.encoder(x_embed) # (B,T,n_embed)

        # Decoder: Case distinction for teacher forcing during training 

        # TODO: Initialize with last value of to predict column of x  
        #x_dec = torch.rand((B,1,1)) # (B,1,1)
        x_dec = x[:,-12:,-1:] # (B,1,1)

        if y_true != None:
            # y_true is (B, h)
            
            y_true = y_true[:,:,None]

            x_dec = torch.cat((x_dec, y_true), dim=-2) # (B, h+1, 1)
            x_dec = self.embed2(x_dec) # (B, h+1, 1) --> (B, h+1, n_emebed)
            for block in self.decoder:
                    x_dec = block(x_dec, x_enc)
            x_dec = self.linear(x_dec) # (B,h+1,n_emebd) --> (B,h+1,1)

            # First element is t+1, last element is t+h+1 thus it needs to be removed
            out = x_dec[:,11:-1,-1] # (B, h+1, 1) --> (B, h, 1)
                
            
        else:
            # TODO: It worked best so far to refeed the embedded outputs in the for loop
            
            for i in range(self.horizon):
                x_dec_old = x_dec # (B, 1+i, 1)

                x_dec = self.embed2(x_dec_old) # (B,1+i, n_embed)
                for block in self.decoder:
                    x_dec = block(x_dec, x_enc)
                # x_dec is (B,1+i,n_embed)
                x_dec = x_dec[:,-1:,:] # (B,1,n_embed)
                x_dec = self.linear(x_dec) # (B,1,1)

                x_dec = torch.cat((x_dec_old, x_dec), dim=-2) # (B, 2+i, 1) --> (B, h+1, 1) after loop
                
                # First element is the initialization t, thus it needs to be removed
                out = x_dec[:,12:,-1] # (B, h+1, 1) --> (B, h)
        if False:
            for i in range(self.horizon):
                x_dec_old = x_dec # (B, 1+i, 1)

                x_dec = self.embed2(x_dec_old) # (B,1+i, n_embed)
                for block in self.decoder:
                    x_dec = block(x_dec, x_enc)
                # x_dec is (B,1+i,n_embed)
                x_dec = x_dec[:,-1:,:] # (B,1,n_embed)
                x_dec = self.linear(x_dec) # (B,1,1)

                x_dec = torch.cat((x_dec_old, x_dec), dim=-2) # (B, 2+i, 1) --> (B, h+1, 1) after loop
                
                # First element is the initialization t, thus it needs to be removed
                out = x_dec[:,12:,-1] # (B, h+1, 1) --> (B, h)


        return out
    
