## Scratch_DiT

This repository provides a **pure PyTorch implementation** of the paper  
[Scaling Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).

The LDM VAE has been ported, and the diffusion samplers are implemented from scratch, making the project fully independent of the diffusers library.  
The DiT architecture is also implemented directly in PyTorch, without relying on timm.

### Configurations in Scope
```bash
DiT_models = {
 #Official arhcitecture DiT_S2
 "DiT-S2": DiT_d12_h6_p2,

 #Depth
 "DiT-d4-h6-p2": DiT_d4_h6_p2,
 "DiT-d6-h6-p2": DiT_d6_h6_p2, 
 "DiT-d8-h6-p2": DiT_d8_h6_p2,
 
 # Heads    
 "DiT-d6-h1-p2": DiT_d6_h1_p2,     
 "DiT-d6-h2-p2": DiT_d6_h2_p2,    
 "DiT-d6-h4-p2": DiT_d6_h4_p2,    
 
 # Patch    
 "DiT-d6-h6-p4": DiT_d6_h6_p4,
 "DiT-d6-h6-p8": DiT_d6_h6_p8

 # Additional models
 "DiT-d6-h1-p2-h768": DiT_d6_h1_p2_h768,
 "DiT-d6-h6-p2-h768": DiT_d6_h6_p2_h768,
 "DiT-d8-h1-p2": DiT_d8_h1_p2,
 }
```
