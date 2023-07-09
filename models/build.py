import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.optim as optim

def build(args, cfg):
    print("Building model: {}".format(args.model))
    if args.model == 'CLFDR':
        from CLFDR import clfdr
        model = clfdr.CLFDR(args)
        
        optim_params = []
        if 'vision' in args.modality_list:
            optim_params.append({'params': model.vision_resnet50.parameters(), 'lr': args.lr*1e-1, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if 'touch' in args.modality_list:
            optim_params.append({'params': model.touch_resnet50.parameters(), 'lr': args.lr*1e-1, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_touch.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        if 'audio' in args.modality_list:
            optim_params.append({'params': model.vggish.parameters(), 'lr': args.lr*1e-1, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_audio.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        
        optim_params.append({'params': model.pointnet2.parameters(), 'lr': args.lr*1e-1, 'weight_decay': args.weight_decay*1e-1})
        optim_params.append({'params': model.mlp_point_cloud.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optim_params.append({'params': model.disentangle_weight, 'lr': args.lr, 'weight_decay': args.weight_decay})
        optim_params.append({'params': model.point_cloud_recon_decoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optim_params.append({'params': model.decoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
    
        optimizer = optim.AdamW(optim_params)
    elif args.model == 'CLR':
        from CLR import clr
        model = clr.CLR(args)
        
        optim_params = []
        if 'vision' in args.modality_list:
            optim_params.append({'params': model.vision_resnet18.parameters(), 'lr': args.lr*1e-1, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_vision.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
            if 'vision_checkpoint' in cfg.keys():
                print(f"loading vision ckpt from {cfg.vision_checkpoint}")
                vision_state_dict = torch.load(cfg.vision_checkpoint, map_location='cpu')
                vision_state_dict = {k: v for k, v in vision_state_dict.items() if 'vision_resnet18' in k}
                model.load_state_dict(vision_state_dict, strict=False)
        if 'touch' in args.modality_list:
            optim_params.append({'params': model.touch_resnet18.parameters(), 'lr': args.lr*1e-3, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_touch.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay})
            if 'touch_checkpoint' in cfg.keys():
                print(f"loading touch ckpt from {cfg.touch_checkpoint}")
                touch_state_dict = torch.load(cfg.touch_checkpoint, map_location='cpu')
                touch_state_dict = {k: v for k, v in touch_state_dict.items() if 'touch_resnet18' in k}
                model.load_state_dict(touch_state_dict, strict=False)
        if 'audio' in args.modality_list:
            optim_params.append({'params': model.audio_resnet18.parameters(), 'lr': args.lr*1e-3, 'weight_decay': args.weight_decay*1e-1})
            optim_params.append({'params': model.mlp_audio.parameters(), 'lr': args.lr*1e-2, 'weight_decay': args.weight_decay})
            if 'audio_checkpoint' in cfg.keys():
                print(f"loading audio ckpt from {cfg.audio_checkpoint}")
                audio_state_dict = torch.load(cfg.audio_checkpoint, map_location='cpu')
                audio_state_dict = {k: v for k, v in audio_state_dict.items() if 'audio_resnet18' in k}
                model.load_state_dict(audio_state_dict, strict=False)
        
        optim_params.append({'params': model.pointnet2.parameters(), 'lr': args.lr*1e-1, 'weight_decay': args.weight_decay*1e-1})
        optim_params.append({'params': model.mlp_point_cloud.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        optim_params.append({'params': model.decoder.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
        
        optimizer = optim.AdamW(optim_params)
    
    return model, optimizer