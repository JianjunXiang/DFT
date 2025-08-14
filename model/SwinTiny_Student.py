import torch
from torch import nn
from einops import rearrange
import argparse
import yaml
from thop import profile
import timm
from model.swin_transformer import swin_tiny_patch4_window7

class LDGA_module(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=768, dropout=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def forward(self, fq, fs):
        fs = self.norm2(fs)
        fqf = self.norm1(fq)
        fqf, self.attn_map = self.multihead_attn(query=fqf, key=fs, value=fs)
        fq = fq + self.dropout2(fqf)
        fqf = self.norm3(fq)
        fqf = self.linear2(self.dropout(self.activation(self.linear1(fqf))))
        fq = fq + self.dropout3(fqf)
        return fq

##################################
class SWTIQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = swin_tiny_patch4_window7(config, pretrained=True)

        self.s_conv0 = nn.Sequential(
            nn.Conv1d(config['o_channel'] // 4, config['o_channel'] // 4, 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'] // 4, config['o_channel'] // 4, 1, 1)
        )

        self.q_conv0 = nn.Sequential(
            nn.Conv1d(config['o_channel'] // 4, config['o_channel'] // 4, 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'] // 4, config['o_channel'] // 4, 1, 1)
        )

        self.s_conv1 = nn.Sequential(
            nn.Conv1d(config['o_channel'] // 2, config['o_channel'] // 2, 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'] // 2, config['o_channel'] // 2, 1, 1)
        )

        self.q_conv1 = nn.Sequential(
            nn.Conv1d(config['o_channel'] // 2, config['o_channel'] // 2, 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'] // 2, config['o_channel'] // 2, 1, 1)
        )

        self.s_conv2 = nn.Sequential(
            nn.Conv1d(config['o_channel'], config['o_channel'] , 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1)
        )

        self.q_conv2 = nn.Sequential(
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1)
        )

        self.s_conv3 = nn.Sequential(
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1)
        )

        self.q_conv3 = nn.Sequential(
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Conv1d(config['o_channel'], config['o_channel'], 1, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(49)
        self.all_channel = config['o_channel'] * 2 + config['o_channel'] // 2 + config['o_channel'] // 4
        self.q_fc = nn.Sequential(
                    nn.Linear(self.all_channel, config['o_channel']),
                    nn.GELU(),
                    nn.Dropout(p=config['drop_ratio']),
                    nn.Linear(config['o_channel'], config['o_channel']),
                )
        self.s_fc = nn.Sequential(
                    nn.Linear(self.all_channel, config['o_channel']),
                    nn.GELU(),
                    nn.Dropout(p=config['drop_ratio']),
                    nn.Linear(config['o_channel'], config['o_channel']),
                )

        self.IQA_head = nn.Sequential(
            nn.Linear(config['o_channel'], config['h_channel']),
            nn.GELU(),
            nn.Dropout(p=config['drop_ratio']),
            nn.Linear(config['h_channel'], 1)
        )

        self.ldga = LDGA_module(d_model=config['o_channel'], nhead=6)

    def forward(self, x):
        _, o_s = self.backbone.forward_features(x)
        feat_s0 = self.s_conv0(o_s[0])
        feat_q0 = self.q_conv0(o_s[0])
        #
        feat_s1 = self.s_conv1(o_s[1])
        feat_q1 = self.q_conv1(o_s[1])
        #
        feat_s2 = self.s_conv2(o_s[2])
        feat_q2 = self.q_conv2(o_s[2])
        #
        feat_s3 = self.s_conv3(o_s[3])
        feat_q3 = self.q_conv3(o_s[3])

        f_s0 = self.avgpool(feat_s0).transpose(1, 2)
        f_q0 = self.avgpool(feat_q0).transpose(1, 2)
        f_s1 = self.avgpool(feat_s1).transpose(1, 2)
        f_q1 = self.avgpool(feat_q1).transpose(1, 2)
        f_s2 = feat_s2.transpose(1, 2)
        f_q2 = feat_q2.transpose(1, 2)
        f_s3 = feat_s3.transpose(1, 2)
        f_q3 = feat_q3.transpose(1, 2)

        feat_q = self.q_fc(torch.cat([f_q0, f_q1, f_q2, f_q3], -1))
        feat_s = self.s_fc(torch.cat([f_s0, f_s1, f_s2, f_s3], -1))
        feat_qs = self.ldga(feat_q, feat_s)

        score = self.IQA_head(feat_qs)
        pred_s = score.mean(-2)
        return [feat_s0, feat_s1, feat_s2, feat_s3], pred_s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swin-Transformer for BIQA in KonIQ dataset.')
    parser.add_argument(
        '-o', '--opt',
        default='./option/train_on_koniq.yaml',
        help='Configuration file'
    )
    args = parser.parse_args()
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    args = parser.parse_args()
    model = SWTIQA(config)
    input_tensor = torch.randn(5, 3, 224, 224)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"Total FLOPs: {flops:,}")
    print(f"Total Parameters: {params:,}")