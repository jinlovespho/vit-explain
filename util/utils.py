import torch 
from torchvision import datasets, transforms
import torch.nn as nn
from functools import partial


def get_model(args):
    if args.model_name == 'mae':
        from models.mae.models_mae import MaskedAutoencoderViT
        model = MaskedAutoencoderViT(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
    elif args.model_name == 'deit':
        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        
    elif args.model_name == '':
        pass 
        
    return model

def get_dataset(args):
    if args.dataset == 'imagenet':
        trans = transforms.Compose([ transforms.Resize(size=(224,224)), transforms.ToTensor() ])
        train_ds = datasets.ImageNet(root='/mnt/ssd2/dataset/ImageNet2012', split='train', transform=trans)
        val_ds = datasets.ImageNet(root='/mnt/ssd2/dataset/ImageNet2012', split='val', transform=trans)
    
    elif args.dataset == '':
        pass 
    
    return train_ds, val_ds 

def get_experiment_name(args):
    experiment_name = f"{args.experiment_memo}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name
