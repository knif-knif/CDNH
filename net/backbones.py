import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import warnings
import os
import timm
import kornia.augmentation as Kg
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import torch

class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseNet, self).__init__()

        self.features_size = 0

    def forward(self, x):
        raise NotImplementedError('please implement `forward`')

class Augmentation(nn.Module):
    def __init__(self, org_size, Aw=1.0):
        super(Augmentation, self).__init__()
        self.gk = int(org_size*0.1)
        if self.gk%2==0:
            self.gk += 1
        self.Aug = nn.Sequential(
        Kg.RandomResizedCrop(size=(org_size, org_size), p=1.0*Aw),
        Kg.RandomHorizontalFlip(p=0.5*Aw),
        Kg.ColorJitter(brightness=0.4, contrast=0.8, saturation=0.8, hue=0.2, p=0.8*Aw),
        Kg.RandomGrayscale(p=0.2*Aw),
        Kg.RandomGaussianBlur((self.gk, self.gk), (0.1, 2.0), p=0.5*Aw))

    def forward(self, x):
        return self.Aug(x)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()        
        self.F = nn.Sequential(*list(models.alexnet(pretrained=True).features))
        self.Pool = nn.AdaptiveAvgPool2d((6,6))
        self.C = nn.Sequential(*list(models.alexnet(pretrained=True).classifier[:-1]))
    def forward(self, x):
        x = self.F(x)
        x = self.Pool(x)
        x = T.flatten(x, 1)
        x = self.C(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.pretrained = models.resnet50(pretrained=True)
        self.children_list = []
        for n,c in self.pretrained.named_children():
            self.children_list.append(c)
            if n == 'avgpool':
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
        
    def forward(self,x):
        x = self.net(x)
        x = T.flatten(x, 1)
        return x

class ViT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class DeiT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        cls_token = self.pm.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat((cls_token, self.pm.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pm.pos_drop(x + self.pm.pos_embed)
        x = self.pm.blocks(x)
        x = self.pm.norm(x)
        return x[:, 0]

class SwinT(nn.Module):
    def __init__(self, pretrained_name):
        super().__init__()
        self.pm = timm.create_model(pretrained_name, pretrained=True)
    def forward(self, x):
        x = self.pm.patch_embed(x)
        if self.pm.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pm.pos_drop(x)
        x = self.pm.layers(x)
        x = self.pm.norm(x)  # B L C
        x = self.pm.avgpool(x.transpose(1, 2))  # B C 1
        x = T.flatten(x, 1)
        return x

import collections.abc
import math
import os
from itertools import repeat

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
ROOTDIR = os.environ.get('ROOTDIR', '.')


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ViTBase(BaseNet):
    name = 'vit_base_patch16_224'

    def __init__(self, pretrained=True, **kwargs):
        super(ViTBase, self).__init__()

        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained,
                                  drop_rate=kwargs.get('drop_rate', 0.),
                                  attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
                                  drop_path_rate=kwargs.get('drop_path_rate', 0.))

        self.embed_dim = model.embed_dim
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        self.norm = model.norm
        self.pre_logits = nn.Identity()
        self.head = model.head

        self.features_size = model.head.in_features
        self.pool_method = kwargs.get('pool_method', 'cls_token')
        # print('pool_method:', self.pool_method)

        self.replace_patch_embed()

    def replace_patch_embed(self):
        # original vit cannot support other input sizes
        patch_embed = PatchEmbed(self.patch_embed.img_size,
                                 self.patch_embed.patch_size,
                                 3,
                                 self.embed_dim)
        patch_embed.load_state_dict(self.patch_embed.state_dict())
        self.patch_embed = patch_embed

    def interpolate_pos_embedding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, with_feat_map=False):
        b, _, h, w = x.size()
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.interpolate_pos_embedding(x, w, h))
        x = self.blocks(x)
        x = self.norm(x)

        if with_feat_map:
            nh = h // self.patch_embed.patch_size[0]
            nw = w // self.patch_embed.patch_size[1]

            return self.pre_logits(x[:, 0]), x[:, 1:].reshape(b, nh, nw, -1).permute(0, 3, 1, 2)
        else:
            if self.pool_method == 'cls_token':
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 1:, :].mean(dim=1)

    def classify(self, x):
        x = self.forward(x)
        x = self.head(x)
        return x


class ViTTiny(ViTBase):
    name = 'vit_tiny_patch16_224'


class ViTSmall(ViTBase):
    name = 'vit_small_patch16_224'


class ViTBaseDino(ViTBase):
    name = 'vit_base_patch16_224_dino'
    filename = 'dino_vitbase16_pretrain.pth'

    def __init__(self, pretrained=True, **kwargs):
        super(ViTBase, self).__init__(**kwargs)

        try:
            model = timm.create_model(self.name, pretrained=pretrained,
                                      drop_rate=kwargs.get('drop_rate', 0.),
                                      attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
                                      drop_path_rate=kwargs.get('drop_path_rate', 0.))
        except:
            model = timm.create_model(self.name.replace('_dino', ''),
                                      drop_rate=kwargs.get('drop_rate', 0.),
                                      attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
                                      drop_path_rate=kwargs.get('drop_path_rate', 0.))
            if pretrained:
                model.load_state_dict(torch.load(f'{ROOTDIR}/pretrained_models/dino/{self.filename}'),
                                      strict=False)

        self.embed_dim = model.embed_dim
        self.num_tokens = model.num_tokens
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        self.norm = model.norm
        self.pre_logits = model.pre_logits
        self.head = model.head

        self.features_size = model.head.in_features
        self.features_size = model.head.in_features
        self.pool_method = kwargs.get('pool_method', 'cls_token')
        print('pool_method:', self.pool_method)

        self.replace_patch_embed()


class ViTSmallDino(ViTBaseDino):
    name = 'vit_small_patch16_224_dino'
    filename = 'dino_deitsmall16_pretrain.pth'


class ViTBaseMAE(ViTBase):
    name = 'vit_base_patch16_224'
    filename = 'mae_visualize_vit_base.pth'

    def __init__(self, pretrained=True, **kwargs):
        super().__init__(pretrained=False, **kwargs)

        if pretrained:
            sd = torch.load(f'{ROOTDIR}/pretrained_models/mae/{self.filename}')['model']
            print(self.load_state_dict(sd, strict=False))


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _attn_forward(attn_module, x):
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)
    attn = attn_module.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x, attn


@torch.no_grad()
def get_attention_and_outputs(vit_model: ViTBase, x: torch.Tensor):
    outputs = {}

    b, _, h, w = x.size()
    x = vit_model.patch_embed(x)
    outputs['patch_embed'] = x

    cls_token = vit_model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)

    x = vit_model.pos_drop(x + vit_model.interpolate_pos_embedding(x, w, h))
    outputs['pos_embed'] = x

    for bidx, block in enumerate(vit_model.blocks):
        z, attn = _attn_forward(block.attn, block.norm1(x))
        x = x + block.drop_path(z)
        x = x + block.drop_path(block.mlp(block.norm2(x)))
        outputs[f'block_{bidx}_attn'] = attn
        outputs[f'block_{bidx}_x'] = x

    x = vit_model.norm(x)
    outputs['output'] = x

    return outputs

class BaseNetE(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        self.nbit = nbit
        self.nclass = nclass

    def count_parameters(self, mode='trainable'):
        if mode == 'trainable':
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif mode == 'non-trainable':
            return sum(p.numel() for p in self.parameters() if not p.requires_grad)
        else:  # all
            return sum(p.numel() for p in self.parameters())

    def finetune_reset(self, *args, **kwargs):
        pass

    def get_backbone(self):
        return self.backbone

    def get_training_modules(self):
        return nn.ModuleDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError('please implement `forward`')

class SDC(BaseNetE):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.hash_fc = nn.Sequential(
            # nn.Linear(self.backbone.features_size, self.backbone.features_size),
            # nn.GELU(),
            # nn.Linear(self.backbone.features_size, self.nbit),
            # nn.BatchNorm1d(self.nbit),
            nn.Linear(self.backbone.features_size, self.nbit),
            nn.LayerNorm(self.nbit),
            nn.Tanh()
        )

    def get_training_modules(self):
        return nn.ModuleDict({'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x)
        v = self.hash_fc(x)
        return x, v



def create_arch(arch, args):
    if arch == 'Vgg':
        Arch = AlexNet()
        args.fc_dim = 4096
    elif arch == 'ResNet':
        Arch = ResNet()
        args.fc_dim = 2048
    elif arch == 'ViT':
        Arch = ViT('vit_base_patch16_224')
        args.fc_dim = 768
    elif arch == 'DeiT':
        Arch = DeiT('deit_base_distilled_patch16_224')
        args.fc_dim = 768
    elif arch == 'SwinT':
        Arch = SwinT('swin_base_patch4_window7_224')
        args.fc_dim = 1024
    else:
        raise ("Wrong dataset name.")
    
    return Arch, args