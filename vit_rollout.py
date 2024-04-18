import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def rollout(attentions, discard_ratio, head_fusion):
    
    # breakpoint()
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():    
        
        # attention heads already fused   
        if len(attentions[0].shape) == 3:
            for attention_heads_fused in attentions:
                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                num_to_discard = int(flat.size(-1)*discard_ratio)
                _, indices = flat.topk(num_to_discard, -1, False)  # largest=False 이기에 topk 반대로 lowerk feel
                indices = indices[indices != 0]     # 
                flat[0, indices] = 0
                
                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
    
        # atteniton heads needs to be fused
        elif len(attentions[0].shape) == 4:   
            for attention in attentions:
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                num_to_discard = int(flat.size(-1)*discard_ratio)
                _, indices = flat.topk(num_to_discard, -1, False)  # largest=False 이기에 topk 반대로 lowerk feel
                indices = indices[indices != 0]     # 
                flat[0, indices] = 0
                
                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
    
    # breakpoint()
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, layer_name1='', layer_name2='', head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        self.lst1=[]
        self.lst2=[]
        for name, module in self.model.named_modules():
            self.lst1.append(name)
            if layer_name1 in name and layer_name2 in name:
                self.lst2.append(name)
                module.register_forward_hook(lambda m,i,o: self.attentions.append(o.detach().cpu()) )

        self.attentions = []

    def __call__(self, input_tensor):
        # breakpoint()
        with torch.no_grad():
            output = self.model(input_tensor)
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)