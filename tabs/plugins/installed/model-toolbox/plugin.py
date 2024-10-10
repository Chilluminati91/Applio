import os, sys
import gradio as gr
import shutil
import torch
from collections import OrderedDict
import torch.nn.functional as F
import re
from assets.i18n.i18n import I18nAuto


# Setup

now_dir = os.getcwd()
sys.path.append(now_dir)
i18n = I18nAuto()

# Layer groups

transform_layers = [
    "enc_p.emb_phone.weight", "enc_p.emb_phone.bias",
    "enc_p.emb_pitch.weight", "enc_p.encoder.attn_layers.0.emb_rel_k",
    "enc_p.encoder.attn_layers.0.emb_rel_v", "enc_p.encoder.attn_layers.0.conv_q.weight",
    "enc_p.encoder.attn_layers.0.conv_q.bias", "enc_p.encoder.attn_layers.0.conv_k.weight",
    "enc_p.encoder.attn_layers.0.conv_k.bias", "enc_p.encoder.attn_layers.0.conv_v.weight",
    "enc_p.encoder.attn_layers.0.conv_v.bias", "enc_p.encoder.attn_layers.0.conv_o.weight",
    "enc_p.encoder.attn_layers.0.conv_o.bias", "enc_p.encoder.attn_layers.1.emb_rel_k",
    "enc_p.encoder.attn_layers.1.emb_rel_v", "enc_p.encoder.attn_layers.1.conv_q.weight",
    "enc_p.encoder.attn_layers.1.conv_q.bias", "enc_p.encoder.attn_layers.1.conv_k.weight",
    "enc_p.encoder.attn_layers.1.conv_k.bias", "enc_p.encoder.attn_layers.1.conv_v.weight",
    "enc_p.encoder.attn_layers.1.conv_v.bias", "enc_p.encoder.attn_layers.1.conv_o.weight",
    "enc_p.encoder.attn_layers.1.conv_o.bias", "enc_p.encoder.attn_layers.2.emb_rel_k",
    "enc_p.encoder.attn_layers.2.emb_rel_v", "enc_p.encoder.attn_layers.2.conv_q.weight",
    "enc_p.encoder.attn_layers.2.conv_q.bias", "enc_p.encoder.attn_layers.2.conv_k.weight",
    "enc_p.encoder.attn_layers.2.conv_k.bias", "enc_p.encoder.attn_layers.2.conv_v.weight",
    "enc_p.encoder.attn_layers.2.conv_v.bias", "enc_p.encoder.attn_layers.2.conv_o.weight",
    "enc_p.encoder.attn_layers.2.conv_o.bias", "enc_p.encoder.attn_layers.3.emb_rel_k",
    "enc_p.encoder.attn_layers.3.emb_rel_v", "enc_p.encoder.attn_layers.3.conv_q.weight",
    "enc_p.encoder.attn_layers.3.conv_q.bias", "enc_p.encoder.attn_layers.3.conv_k.weight",
    "enc_p.encoder.attn_layers.3.conv_k.bias", "enc_p.encoder.attn_layers.3.conv_v.weight",
    "enc_p.encoder.attn_layers.3.conv_v.bias", "enc_p.encoder.attn_layers.3.conv_o.weight",
    "enc_p.encoder.attn_layers.3.conv_o.bias", "enc_p.encoder.attn_layers.4.emb_rel_k",
    "enc_p.encoder.attn_layers.4.emb_rel_v", "enc_p.encoder.attn_layers.4.conv_q.weight",
    "enc_p.encoder.attn_layers.4.conv_q.bias", "enc_p.encoder.attn_layers.4.conv_k.weight",
    "enc_p.encoder.attn_layers.4.conv_k.bias", "enc_p.encoder.attn_layers.4.conv_v.weight",
    "enc_p.encoder.attn_layers.4.conv_v.bias", "enc_p.encoder.attn_layers.4.conv_o.weight",
    "enc_p.encoder.attn_layers.4.conv_o.bias", "enc_p.encoder.attn_layers.5.emb_rel_k",
    "enc_p.encoder.attn_layers.5.emb_rel_v", "enc_p.encoder.attn_layers.5.conv_q.weight",
    "enc_p.encoder.attn_layers.5.conv_q.bias", "enc_p.encoder.attn_layers.5.conv_k.weight",
    "enc_p.encoder.attn_layers.5.conv_k.bias", "enc_p.encoder.attn_layers.5.conv_v.weight",
    "enc_p.encoder.attn_layers.5.conv_v.bias", "enc_p.encoder.attn_layers.5.conv_o.weight",
    "enc_p.encoder.attn_layers.5.conv_o.bias", "enc_p.encoder.norm_layers_1.0.gamma",
    "enc_p.encoder.norm_layers_1.0.beta", "enc_p.encoder.norm_layers_1.1.gamma",
    "enc_p.encoder.norm_layers_1.1.beta", "enc_p.encoder.norm_layers_1.2.gamma",
    "enc_p.encoder.norm_layers_1.2.beta", "enc_p.encoder.norm_layers_1.3.gamma",
    "enc_p.encoder.norm_layers_1.3.beta", "enc_p.encoder.norm_layers_1.4.gamma",
    "enc_p.encoder.norm_layers_1.4.beta", "enc_p.encoder.norm_layers_1.5.gamma",
    "enc_p.encoder.norm_layers_1.5.beta", "enc_p.encoder.ffn_layers.0.conv_1.weight",
    "enc_p.encoder.ffn_layers.0.conv_1.bias", "enc_p.encoder.ffn_layers.0.conv_2.weight",
    "enc_p.encoder.ffn_layers.0.conv_2.bias", "enc_p.encoder.ffn_layers.1.conv_1.weight",
    "enc_p.encoder.ffn_layers.1.conv_1.bias", "enc_p.encoder.ffn_layers.1.conv_2.weight",
    "enc_p.encoder.ffn_layers.1.conv_2.bias", "enc_p.encoder.ffn_layers.2.conv_1.weight",
    "enc_p.encoder.ffn_layers.2.conv_1.bias", "enc_p.encoder.ffn_layers.2.conv_2.weight",
    "enc_p.encoder.ffn_layers.2.conv_2.bias", "enc_p.encoder.ffn_layers.3.conv_1.weight",
    "enc_p.encoder.ffn_layers.3.conv_1.bias", "enc_p.encoder.ffn_layers.3.conv_2.weight",
    "enc_p.encoder.ffn_layers.3.conv_2.bias", "enc_p.encoder.ffn_layers.4.conv_1.weight",
    "enc_p.encoder.ffn_layers.4.conv_1.bias", "enc_p.encoder.ffn_layers.4.conv_2.weight",
    "enc_p.encoder.ffn_layers.4.conv_2.bias", "enc_p.encoder.ffn_layers.5.conv_1.weight",
    "enc_p.encoder.ffn_layers.5.conv_1.bias", "enc_p.encoder.ffn_layers.5.conv_2.weight",
    "enc_p.encoder.ffn_layers.5.conv_2.bias", "enc_p.encoder.norm_layers_2.0.gamma",
    "enc_p.encoder.norm_layers_2.0.beta", "enc_p.encoder.norm_layers_2.1.gamma",
    "enc_p.encoder.norm_layers_2.1.beta", "enc_p.encoder.norm_layers_2.2.gamma",
    "enc_p.encoder.norm_layers_2.2.beta", "enc_p.encoder.norm_layers_2.3.gamma",
    "enc_p.encoder.norm_layers_2.3.beta", "enc_p.encoder.norm_layers_2.4.gamma",
    "enc_p.encoder.norm_layers_2.4.beta", "enc_p.encoder.norm_layers_2.5.gamma",
    "enc_p.encoder.norm_layers_2.5.beta", "enc_p.proj.weight",
    "enc_p.proj.bias", "dec.m_source.l_linear.weight",
    "dec.m_source.l_linear.bias", "dec.resblocks.0.convs1.0.bias",
    "dec.resblocks.0.convs1.0.weight_g", "dec.resblocks.0.convs1.0.weight_v",
    "dec.resblocks.0.convs1.1.bias", "dec.resblocks.0.convs1.1.weight_g",
    "dec.resblocks.0.convs1.1.weight_v", "dec.resblocks.0.convs1.2.bias",
    "dec.resblocks.0.convs1.2.weight_g", "dec.resblocks.0.convs1.2.weight_v",
    "dec.resblocks.0.convs2.0.bias", "dec.resblocks.0.convs2.0.weight_g",
    "dec.resblocks.0.convs2.0.weight_v", "dec.resblocks.0.convs2.1.bias",
    "dec.resblocks.0.convs2.1.weight_g", "dec.resblocks.0.convs2.1.weight_v",
    "dec.resblocks.0.convs2.2.bias", "dec.resblocks.0.convs2.2.weight_g",
    "dec.resblocks.0.convs2.2.weight_v", "dec.resblocks.1.convs1.0.bias",
    "dec.resblocks.1.convs1.0.weight_g", "dec.resblocks.1.convs1.0.weight_v",
    "dec.resblocks.1.convs1.1.bias", "dec.resblocks.1.convs1.1.weight_g",
    "dec.resblocks.1.convs1.1.weight_v", "dec.resblocks.1.convs1.2.bias",
    "dec.resblocks.1.convs1.2.weight_g", "dec.resblocks.1.convs1.2.weight_v",
    "dec.resblocks.1.convs2.0.bias", "dec.resblocks.1.convs2.0.weight_g",
    "dec.resblocks.1.convs2.0.weight_v", "dec.resblocks.1.convs2.1.bias",
    "dec.resblocks.1.convs2.1.weight_g", "dec.resblocks.1.convs2.1.weight_v",
    "dec.resblocks.1.convs2.2.bias", "dec.resblocks.1.convs2.2.weight_g",
    "dec.resblocks.1.convs2.2.weight_v", "dec.resblocks.2.convs1.0.bias",
    "dec.resblocks.2.convs1.0.weight_g", "dec.resblocks.2.convs1.0.weight_v",
    "dec.resblocks.2.convs1.1.bias", "dec.resblocks.2.convs1.1.weight_g",
    "dec.resblocks.2.convs1.1.weight_v", "dec.resblocks.2.convs1.2.bias",
    "dec.resblocks.2.convs1.2.weight_g", "dec.resblocks.2.convs1.2.weight_v",
    "dec.resblocks.2.convs2.0.bias", "dec.resblocks.2.convs2.0.weight_g",
    "dec.resblocks.2.convs2.0.weight_v", "dec.resblocks.2.convs2.1.bias",
    "dec.resblocks.2.convs2.1.weight_g", "dec.resblocks.2.convs2.1.weight_v",
    "dec.resblocks.2.convs2.2.bias", "dec.resblocks.2.convs2.2.weight_g",
    "dec.resblocks.2.convs2.2.weight_v", "dec.resblocks.3.convs1.0.bias",
    "dec.resblocks.3.convs1.0.weight_g", "dec.resblocks.3.convs1.0.weight_v",
    "dec.resblocks.3.convs1.1.bias", "dec.resblocks.3.convs1.1.weight_g",
    "dec.resblocks.3.convs1.1.weight_v", "dec.resblocks.3.convs1.2.bias",
    "dec.resblocks.3.convs1.2.weight_g", "dec.resblocks.3.convs1.2.weight_v",
    "dec.resblocks.3.convs2.0.bias", "dec.resblocks.3.convs2.0.weight_g",
    "dec.resblocks.3.convs2.0.weight_v", "dec.resblocks.3.convs2.1.bias",
    "dec.resblocks.3.convs2.1.weight_g", "dec.resblocks.3.convs2.1.weight_v",
    "dec.resblocks.3.convs2.2.bias", "dec.resblocks.3.convs2.2.weight_g",
    "dec.resblocks.3.convs2.2.weight_v", "dec.resblocks.4.convs1.0.bias",
    "dec.resblocks.4.convs1.0.weight_g", "dec.resblocks.4.convs1.0.weight_v",
    "dec.resblocks.4.convs1.1.bias", "dec.resblocks.4.convs1.1.weight_g",
    "dec.resblocks.4.convs1.1.weight_v", "dec.resblocks.4.convs1.2.bias",
    "dec.resblocks.4.convs1.2.weight_g", "dec.resblocks.4.convs1.2.weight_v",
    "dec.resblocks.4.convs2.0.bias", "dec.resblocks.4.convs2.0.weight_g",
    "dec.resblocks.4.convs2.0.weight_v", "dec.resblocks.4.convs2.1.bias",
    "dec.resblocks.4.convs2.1.weight_g", "dec.resblocks.4.convs2.1.weight_v",
    "dec.resblocks.4.convs2.2.bias", "dec.resblocks.4.convs2.2.weight_g",
    "dec.resblocks.4.convs2.2.weight_v", "dec.resblocks.5.convs1.0.bias",
    "dec.resblocks.5.convs1.0.weight_g", "dec.resblocks.5.convs1.0.weight_v",
    "dec.resblocks.5.convs1.1.bias", "dec.resblocks.5.convs1.1.weight_g",
    "dec.resblocks.5.convs1.1.weight_v", "dec.resblocks.5.convs1.2.bias",
    "dec.resblocks.5.convs1.2.weight_g", "dec.resblocks.5.convs1.2.weight_v",
    "dec.resblocks.5.convs2.0.bias", "dec.resblocks.5.convs2.0.weight_g",
    "dec.resblocks.5.convs2.0.weight_v", "dec.resblocks.5.convs2.1.bias",
    "dec.resblocks.5.convs2.1.weight_g", "dec.resblocks.5.convs2.1.weight_v",
    "dec.resblocks.5.convs2.2.bias", "dec.resblocks.5.convs2.2.weight_g",
    "dec.resblocks.5.convs2.2.weight_v", "dec.resblocks.6.convs1.0.bias",
    "dec.resblocks.6.convs1.0.weight_g", "dec.resblocks.6.convs1.0.weight_v",
    "dec.resblocks.6.convs1.1.bias", "dec.resblocks.6.convs1.1.weight_g",
    "dec.resblocks.6.convs1.1.weight_v", "dec.resblocks.6.convs1.2.bias",
    "dec.resblocks.6.convs1.2.weight_g", "dec.resblocks.6.convs1.2.weight_v",
    "dec.resblocks.6.convs2.0.bias", "dec.resblocks.6.convs2.0.weight_g",
    "dec.resblocks.6.convs2.0.weight_v", "dec.resblocks.6.convs2.1.bias",
    "dec.resblocks.6.convs2.1.weight_g", "dec.resblocks.6.convs2.1.weight_v",
    "dec.resblocks.6.convs2.2.bias", "dec.resblocks.6.convs2.2.weight_g",
    "dec.resblocks.6.convs2.2.weight_v", "dec.resblocks.7.convs1.0.bias",
    "dec.resblocks.7.convs1.0.weight_g", "dec.resblocks.7.convs1.0.weight_v",
    "dec.resblocks.7.convs1.1.bias", "dec.resblocks.7.convs1.1.weight_g",
    "dec.resblocks.7.convs1.1.weight_v", "dec.resblocks.7.convs1.2.bias",
    "dec.resblocks.7.convs1.2.weight_g", "dec.resblocks.7.convs1.2.weight_v",
    "dec.resblocks.7.convs2.0.bias", "dec.resblocks.7.convs2.0.weight_g",
    "dec.resblocks.7.convs2.0.weight_v", "dec.resblocks.7.convs2.1.bias",
    "dec.resblocks.7.convs2.1.weight_g", "dec.resblocks.7.convs2.1.weight_v",
    "dec.resblocks.7.convs2.2.bias", "dec.resblocks.7.convs2.2.weight_g",
    "dec.resblocks.7.convs2.2.weight_v", "dec.resblocks.8.convs1.0.bias",
    "dec.resblocks.8.convs1.0.weight_g", "dec.resblocks.8.convs1.0.weight_v",
    "dec.resblocks.8.convs1.1.bias", "dec.resblocks.8.convs1.1.weight_g",
    "dec.resblocks.8.convs1.1.weight_v", "dec.resblocks.8.convs1.2.bias",
    "dec.resblocks.8.convs1.2.weight_g", "dec.resblocks.8.convs1.2.weight_v",
    "dec.resblocks.8.convs2.0.bias", "dec.resblocks.8.convs2.0.weight_g",
    "dec.resblocks.8.convs2.0.weight_v", "dec.resblocks.8.convs2.1.bias",
    "dec.resblocks.8.convs2.1.weight_g", "dec.resblocks.8.convs2.1.weight_v",
    "dec.resblocks.8.convs2.2.bias", "dec.resblocks.8.convs2.2.weight_g",
    "dec.resblocks.8.convs2.2.weight_v", "dec.resblocks.9.convs1.0.bias",
    "dec.resblocks.9.convs1.0.weight_g", "dec.resblocks.9.convs1.0.weight_v",
    "dec.resblocks.9.convs1.1.bias", "dec.resblocks.9.convs1.1.weight_g",
    "dec.resblocks.9.convs1.1.weight_v", "dec.resblocks.9.convs1.2.bias",
    "dec.resblocks.9.convs1.2.weight_g", "dec.resblocks.9.convs1.2.weight_v",
    "dec.resblocks.9.convs2.0.bias", "dec.resblocks.9.convs2.0.weight_g",
    "dec.resblocks.9.convs2.0.weight_v", "dec.resblocks.9.convs2.1.bias",
    "dec.resblocks.9.convs2.1.weight_g", "dec.resblocks.9.convs2.1.weight_v",
    "dec.resblocks.9.convs2.2.bias", "dec.resblocks.9.convs2.2.weight_g",
    "dec.resblocks.9.convs2.2.weight_v", "dec.resblocks.10.convs1.0.bias",
    "dec.resblocks.10.convs1.0.weight_g", "dec.resblocks.10.convs1.0.weight_v",
    "dec.resblocks.10.convs1.1.bias", "dec.resblocks.10.convs1.1.weight_g",
    "dec.resblocks.10.convs1.1.weight_v", "dec.resblocks.10.convs1.2.bias",
    "dec.resblocks.10.convs1.2.weight_g", "dec.resblocks.10.convs1.2.weight_v",
    "dec.resblocks.10.convs2.0.bias", "dec.resblocks.10.convs2.0.weight_g",
    "dec.resblocks.10.convs2.0.weight_v", "dec.resblocks.10.convs2.1.bias",
    "dec.resblocks.10.convs2.1.weight_g", "dec.resblocks.10.convs2.1.weight_v",
    "dec.resblocks.10.convs2.2.bias", "dec.resblocks.10.convs2.2.weight_g",
    "dec.resblocks.10.convs2.2.weight_v", "dec.resblocks.11.convs1.0.bias",
    "dec.resblocks.11.convs1.0.weight_g", "dec.resblocks.11.convs1.0.weight_v",
    "dec.resblocks.11.convs1.1.bias", "dec.resblocks.11.convs1.1.weight_g",
    "dec.resblocks.11.convs1.1.weight_v", "dec.resblocks.11.convs1.2.bias",
    "dec.resblocks.11.convs1.2.weight_g", "dec.resblocks.11.convs1.2.weight_v",
    "dec.resblocks.11.convs2.0.bias", "dec.resblocks.11.convs2.0.weight_g",
    "dec.resblocks.11.convs2.0.weight_v", "dec.resblocks.11.convs2.1.bias",
    "dec.resblocks.11.convs2.1.weight_g", "dec.resblocks.11.convs2.1.weight_v",
    "dec.resblocks.11.convs2.2.bias", "dec.resblocks.11.convs2.2.weight_g",
    "dec.resblocks.11.convs2.2.weight_v", "flow.flows.0.pre.weight",
    "flow.flows.0.pre.bias", "flow.flows.0.enc.in_layers.0.bias",
    "flow.flows.0.enc.in_layers.0.weight_g", "flow.flows.0.enc.in_layers.0.weight_v",
    "flow.flows.0.enc.in_layers.1.bias", "flow.flows.0.enc.in_layers.1.weight_g",
    "flow.flows.0.enc.in_layers.1.weight_v", "flow.flows.0.enc.in_layers.2.bias",
    "flow.flows.0.enc.in_layers.2.weight_g", "flow.flows.0.enc.in_layers.2.weight_v",
    "flow.flows.0.enc.res_skip_layers.0.bias", "flow.flows.0.enc.res_skip_layers.0.weight_g",
    "flow.flows.0.enc.res_skip_layers.0.weight_v", "flow.flows.0.enc.res_skip_layers.1.bias",
    "flow.flows.0.enc.res_skip_layers.1.weight_g", "flow.flows.0.enc.res_skip_layers.1.weight_v",
    "flow.flows.0.enc.res_skip_layers.2.bias", "flow.flows.0.enc.res_skip_layers.2.weight_g",
    "flow.flows.0.enc.res_skip_layers.2.weight_v", "flow.flows.0.enc.cond_layer.bias",
    "flow.flows.0.enc.cond_layer.weight_g", "flow.flows.0.enc.cond_layer.weight_v",
    "flow.flows.0.post.weight", "flow.flows.0.post.bias",
    "flow.flows.2.pre.weight", "flow.flows.2.pre.bias",
    "flow.flows.2.enc.in_layers.0.bias", "flow.flows.2.enc.in_layers.0.weight_g",
    "flow.flows.2.enc.in_layers.0.weight_v", "flow.flows.2.enc.in_layers.1.bias",
    "flow.flows.2.enc.in_layers.1.weight_g", "flow.flows.2.enc.in_layers.1.weight_v",
    "flow.flows.2.enc.in_layers.2.bias", "flow.flows.2.enc.in_layers.2.weight_g",
    "flow.flows.2.enc.in_layers.2.weight_v", "flow.flows.2.enc.res_skip_layers.0.bias",
    "flow.flows.2.enc.res_skip_layers.0.weight_g", "flow.flows.2.enc.res_skip_layers.0.weight_v",
    "flow.flows.2.enc.res_skip_layers.1.bias", "flow.flows.2.enc.res_skip_layers.1.weight_g",
    "flow.flows.2.enc.res_skip_layers.1.weight_v", "flow.flows.2.enc.res_skip_layers.2.bias",
    "flow.flows.2.enc.res_skip_layers.2.weight_g", "flow.flows.2.enc.res_skip_layers.2.weight_v",
    "flow.flows.2.enc.cond_layer.bias", "flow.flows.2.enc.cond_layer.weight_g",
    "flow.flows.2.enc.cond_layer.weight_v", "flow.flows.2.post.weight",
    "flow.flows.2.post.bias", "flow.flows.4.pre.weight",
    "flow.flows.4.pre.bias", "flow.flows.4.enc.in_layers.0.bias",
    "flow.flows.4.enc.in_layers.0.weight_g", "flow.flows.4.enc.in_layers.0.weight_v",
    "flow.flows.4.enc.in_layers.1.bias", "flow.flows.4.enc.in_layers.1.weight_g",
    "flow.flows.4.enc.in_layers.1.weight_v", "flow.flows.4.enc.in_layers.2.bias",
    "flow.flows.4.enc.in_layers.2.weight_g", "flow.flows.4.enc.in_layers.2.weight_v",
    "flow.flows.4.enc.res_skip_layers.0.bias", "flow.flows.4.enc.res_skip_layers.0.weight_g",
    "flow.flows.4.enc.res_skip_layers.0.weight_v", "flow.flows.4.enc.res_skip_layers.1.bias",
    "flow.flows.4.enc.res_skip_layers.1.weight_g", "flow.flows.4.enc.res_skip_layers.1.weight_v",
    "flow.flows.4.enc.res_skip_layers.2.bias", "flow.flows.4.enc.res_skip_layers.2.weight_g",
    "flow.flows.4.enc.res_skip_layers.2.weight_v", "flow.flows.4.enc.cond_layer.bias",
    "flow.flows.4.enc.cond_layer.weight_g", "flow.flows.4.enc.cond_layer.weight_v",
    "flow.flows.4.post.weight", "flow.flows.4.post.bias",
    "flow.flows.6.pre.weight", "flow.flows.6.pre.bias",
    "flow.flows.6.enc.in_layers.0.bias", "flow.flows.6.enc.in_layers.0.weight_g",
    "flow.flows.6.enc.in_layers.0.weight_v", "flow.flows.6.enc.in_layers.1.bias",
    "flow.flows.6.enc.in_layers.1.weight_g", "flow.flows.6.enc.in_layers.1.weight_v",
    "flow.flows.6.enc.in_layers.2.bias", "flow.flows.6.enc.in_layers.2.weight_g",
    "flow.flows.6.enc.in_layers.2.weight_v", "flow.flows.6.enc.res_skip_layers.0.bias",
    "flow.flows.6.enc.res_skip_layers.0.weight_g", "flow.flows.6.enc.res_skip_layers.0.weight_v",
    "flow.flows.6.enc.res_skip_layers.1.bias", "flow.flows.6.enc.res_skip_layers.1.weight_g",
    "flow.flows.6.enc.res_skip_layers.1.weight_v", "flow.flows.6.enc.res_skip_layers.2.bias",
    "flow.flows.6.enc.res_skip_layers.2.weight_g", "flow.flows.6.enc.res_skip_layers.2.weight_v",
    "flow.flows.6.enc.cond_layer.bias", "flow.flows.6.enc.cond_layer.weight_g",
    "flow.flows.6.enc.cond_layer.weight_v", "flow.flows.6.post.weight",
    "flow.flows.6.post.bias", "emb_g.weight"
]

quality_layers = [
    "dec.ups.0.bias", "dec.ups.0.weight_g",
    "dec.ups.0.weight_v", "dec.ups.1.bias",
    "dec.ups.1.weight_g", "dec.ups.1.weight_v",
    "dec.ups.2.bias", "dec.ups.2.weight_g",
    "dec.ups.2.weight_v", "dec.ups.3.bias",
    "dec.ups.3.weight_g", "dec.ups.3.weight_v",
    "dec.noise_convs.0.weight", "dec.noise_convs.0.bias",
    "dec.noise_convs.1.weight", "dec.noise_convs.1.bias",
    "dec.noise_convs.2.weight", "dec.noise_convs.2.bias",
    "dec.noise_convs.3.weight", "dec.noise_convs.3.bias",
    "dec.conv_post.weight"
]

input_layers = [
    "dec.cond.weight", "dec.cond.bias"
]

input_feature_layers = [
    "dec.conv_pre.weight", "dec.conv_pre.bias"
]

# Helper Functions

def update_model_fusion(dropbox):
    return dropbox, None

def extract(ckpt):
    a = ckpt["model"]
    opt = OrderedDict()
    opt["weight"] = {}
    for key in a.keys():
        if "enc_q" in key:
            continue
        opt["weight"][key] = a[key]
    return opt

def update_checkboxes(selected):
    if "All Layers" in selected:
        # If "All Layers" is selected, uncheck all others
        return ["All Layers"]  # Only keep "All Layers" checked
    return selected  # Otherwise, return the current selections

# Interpolation Functions

def interpolate_models(name, path1, path2, ratio, interpolation_types):
    try:
        # Ensure ratio is a float
        ratio = float(ratio)

        model1_name = os.path.basename(path1)
        model2_name = os.path.basename(path2)
        
        message = f"Model {model1_name} and {model2_name} are merged with alpha {ratio} using {', '.join(interpolation_types)} interpolation."
        ckpt1 = torch.load(path1, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        ckpt2 = torch.load(path2, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Extract model metadata
        cfg = ckpt1["config"]
        cfg_f0 = ckpt1["f0"]
        cfg_version = ckpt1["version"]
        cfg_sr = ckpt1["sr"]

        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]

        # Check model compatibility
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."

        # Initialize the new model
        opt = OrderedDict()
        opt["weight"] = {}

        # Define a mapping for interpolation types to layers
        layer_mapping = {
            "Transform Layers": transform_layers,
            "Quality Layers": quality_layers,
            "Input Layers": input_layers,
            "Input Feature Layers": input_feature_layers,
        }

        # Create a set of layers to interpolate based on selected types
        layers_to_interpolate = set()
        for interp_type in interpolation_types:
            if interp_type == "All Layers":
                layers_to_interpolate = set(ckpt1.keys())  # Merge entire model
                break  # Exit loop if "All Layers" is selected
            elif interp_type in layer_mapping:
                layers_to_interpolate.update(layer_mapping[interp_type])

        # Perform interpolation based on selected layers
        for key in ckpt1.keys():
            vector1 = ckpt1[key].float()
            vector2 = ckpt2[key].float()

            if key == "emb_g.weight" and vector1.shape != vector2.shape:
                min_shape0 = min(vector1.shape[0], vector2.shape[0])
                vector1, vector2 = vector1[:min_shape0], vector2[:min_shape0]

            # Interpolate based on whether the layer is selected for merging
            if key in layers_to_interpolate:
                opt["weight"][key] = (ratio * vector1 + (1 - ratio) * vector2).half()
            else:
                # For non-specified layers, just copy from one model (e.g., ckpt1)
                opt["weight"][key] = vector1.half()

        # Save blended model with original config metadata
        opt["config"] = cfg
        opt["sr"] = cfg_sr
        opt["f0"] = cfg_f0
        opt["version"] = cfg_version
        opt["info"] = message

        save_path = os.path.join("logs", f"{name}.pth")
        torch.save(opt, save_path)
        print(message)
        return message, save_path

    except Exception as error:
        print(f"An error occurred blending the models: {error}")
        return str(error)

# GUI

def applio_plugin():
    gr.Markdown(i18n("## Merge Utilities"))
    gr.Markdown(
        i18n(
            "Different tools to interpolate two models with eachother. Make sure that both models have the same samplerate, this script does not check!"
        )
    )
    with gr.Column():
        model_fusion_name = gr.Textbox(
            label=i18n("Model Name"),
            info=i18n("Name of the new model."),
            value="",
            max_lines=1,
            interactive=True,
            placeholder=i18n("Enter model name"),
        )
        with gr.Row():
            with gr.Column():
                model_fusion_a_dropbox = gr.File(
                    label=i18n("Drag and drop your model here"), type="filepath"
                )
                model_fusion_a = gr.Textbox(
                    label=i18n("Path to Model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter path to model"),
                    info=i18n("You can also use a custom path."),
                )
            with gr.Column():
                model_fusion_b_dropbox = gr.File(
                    label=i18n("Drag and drop your model here"), type="filepath"
                )
                model_fusion_b = gr.Textbox(
                    label=i18n("Path to Model"),
                    value="",
                    interactive=True,
                    placeholder=i18n("Enter path to model"),
                    info=i18n("You can also use a custom path."),
                )
        with gr.Column():
            ratio = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Blend Ratio"),
                value=0.5,
                interactive=True,
                info=i18n(
                    "The weight of model a in all functions. At 0.8 it makes a mix of 80% model1 and 20% model2."
                ),
            )
    with gr.Row():
        with gr.Column():
            interpolation_checkbox_group = gr.CheckboxGroup(
                ["All Layers", 
                 "Transform Layers", 
                 "Quality Layers", 
                 "Input Layers", 
                 "Input Feature Layers"], 
                label="Interpolation Layers", 
                info="Choose the layers to interpolate.", 
                value=[],
                interactive=True,
            )
            model_fusion_button1 = gr.Button(i18n("Blend Models"), variant="primary")
        with gr.Column():
            model_fusion_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            lines=3,
            )

# Buttons

    interpolation_checkbox_group.change(
        fn=update_checkboxes,
        inputs=interpolation_checkbox_group,
        outputs=interpolation_checkbox_group
    )

    model_fusion_button1.click(
        fn=interpolate_models,
        inputs=[
            model_fusion_name,
            model_fusion_a,
            model_fusion_b,
            ratio,
            interpolation_checkbox_group,
        ],
        outputs=[model_fusion_output_info],
    )

    model_fusion_a_dropbox.upload(
        fn=update_model_fusion,
        inputs=model_fusion_a_dropbox,
        outputs=[model_fusion_a, model_fusion_a_dropbox],
    )

    model_fusion_b_dropbox.upload(
        fn=update_model_fusion,
        inputs=model_fusion_b_dropbox,
        outputs=[model_fusion_b, model_fusion_b_dropbox],
    )