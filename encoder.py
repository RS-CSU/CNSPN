import torch.nn as nn

import torch
import torch.nn.functional as F

from feature_extractor import ResNet18
def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet_4(nn.Module):
    def __init__(self, x_input_dim=3, hid_dim=64,shot=1,word2vec_length=300,output_dimension=512,fusion='avg'):
        super().__init__()
        self.fusion = fusion #avg,weight_avg,att,mfb
        self.encoder = ResNet18()
        if word2vec_length<= 512:
            self.word_mapping = nn.Sequential(nn.Linear(word2vec_length,output_dimension))
        else:
            self.word_mapping = nn.Sequential(nn.Linear(word2vec_length,word2vec_length//2),
                                              # nn.BatchNorm1d(word2vec_length//2),
                                              # nn.ReLU(),
                                              nn.Linear(word2vec_length//2,output_dimension))

        self.vis_mapping = nn.Linear(1600, output_dimension) #1600->512
        self.shot = shot


        if self.fusion == 'concat':
            self.Linear_concat = nn.Linear(512*2, 512)

        if self.fusion == 'avg':
            self.Linear_avg = nn.Linear(512, 512)

        if self.fusion == 'weight_avg':
            # self.weight_type = 'non-linear' #加权平均时,得到权重的方式是线性还是非线性
            # self.weight_proj = nn.Linear(512,1)
            self.weight_proj = nn.Sequential(nn.Linear(512,300),
                                             nn.Dropout(p=0.5),
                                             nn.Linear(300,1))
        if self.fusion == 'mfb':
            self.joint_emb_size = 5*512
            self.Linear_imgproj = nn.Linear(512,5 * 512)
            self.Linear_textproj = nn.Linear(512, 5 * 512)



    def forward(self, x, class_name):#forward_support
        way = class_name.size(0)//self.shot
        class_name = class_name[:way]
        x = self.encoder(x)
        # short_x = x
        # x = x.view(x.size(0), -1) #(5,1600)
        # print(x.shape)
        # x = self.vis_mapping(x) #(5,512)

        # x = x/x.norm(dim=-1,keepdim=True) #norm


        if self.shot != 1:
            vis_proto = x.view((self.shot, -1, 512)).mean(dim=0) # get visual prototype
        else:
            vis_proto = x
            # print(vis_proto.shape)
        # print(class_name.shape)
        text_embed = self.word_mapping(class_name) #文本嵌入

        text_embed = text_embed/text_embed.norm(dim=-1,keepdim=True)

        if self.fusion == 'concat':
            vis_text_proto = torch.cat([vis_proto,text_embed],dim=-1)
            vis_text_proto = self.Linear_concat(vis_text_proto)
        elif self.fusion == 'avg':
            vis_text_proto = (vis_proto + text_embed)/2.0
            vis_text_proto = self.Linear_avg(vis_text_proto)
        elif self.fusion == 'weight_avg':
            h = self.weight_proj(text_embed)
            h = h.sigmoid()
            vis_text_proto =h*vis_proto + (1-h)*text_embed
        elif self.fusion == 'mfb':
        # MFB fu
            text_feature = self.Linear_textproj(text_embed)
            img_feature = self.Linear_imgproj(vis_proto)
            il = torch.mul(text_feature,img_feature)
            il = F.dropout(il,0.1,training=self.training)
            il = il.view(-1,1,512,5)
            il = torch.squeeze(torch.sum(il,3))
            il = torch.sqrt(F.relu(il)) - torch.sqrt(F.relu(-il))
            vis_text_proto = F.normalize(il)
        # print(vis_text_proto.shape)

        return vis_text_proto,vis_proto

    def forward_query(self , x):
        x = self.encoder(x)
        # x = x / x.norm(dim=-1, keepdim=True)  # 单纯测试视觉的时候，不需要进行正则。
        # x = x.view(x.size(0), -1)  # (5,1600)
        # x = self.vis_mapping(x)
        # print(x.shape)

        # x = x / x.norm(dim=-1, keepdim=True)
        return x
    def return_score(self,x):
        pass
