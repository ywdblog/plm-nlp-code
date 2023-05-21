from torch import nn
import  torch 

# d_model: 输入和输出的维度 ，nhead: 多头注意力的头数
encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=2)

# src: 输入张量，shape为(src_len, batch, d_model)
# 序列长度 批次 单词个数
src = torch.rand(2,3,4) 
out = encoder_layer(src)
print(out )

# 将多个编码器层组成一个编码器
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
out = transformer_encoder(src)

# 解码器
memory = transformer_encoder(src)
decoder_layer = nn.TransformerDecoderLayer(d_model=4, nhead=2)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
out_part = torch.rand(2, 3, 4)
out = transformer_decoder(out_part, memory)


