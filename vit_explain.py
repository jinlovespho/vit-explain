import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt 

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from util.utils import get_model, get_dataset 
from util.show_utils import show_mask_on_image
from args import get_args


if __name__ == '__main__':
    args = get_args()

    # get model
    model = get_model(args)
    model.eval()
    
    # get data
    train_ds, val_ds = get_dataset(args)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8)
    train_dl_iter = iter(train_dl)
    
    # get data(x) and label(y)
    x, y = next(train_dl_iter)
    
    # GIVEN EXAMPLE
    # img = Image.open('./examples/plane2.png')
    # img = img.resize((224,224))
    # trans = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor() ])
    # x = trans(img).unsqueeze(dim=0)
    
    # get Attention Map
    attn_rollout = VITAttentionRollout(model, layer_name1='', layer_name2='attn_drop', head_fusion=args.head_fusion, discard_ratio=args.discard_ratio)
    attn_map = attn_rollout(x)
    name = f"img_attn_map_{args.head_fusion}_{args.discard_ratio}.png"
    
    # PYTORCH fused_attn=True 로 되어있으면 hook 안걸린다 ! 
    # for block in model.blocks:
    #         block.attn.fused_attn = False

    # visualize attn map and image
    np_img = x.squeeze(dim=0).permute(1,2,0).numpy() * 255 
    np_img = np_img[:,:, [2,1,0] ]
    mask = cv2.resize(attn_map, (np_img.shape[1], np_img.shape[0])) #(h,w)
    mask = show_mask_on_image(np_img, mask)
    cv2.imwrite("img.png", np_img)
    cv2.imwrite(name, mask)
    