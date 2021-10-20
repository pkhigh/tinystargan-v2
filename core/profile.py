import sys
import os
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import torch.nn as nn
from torchprofile import profile_macs

from core.model import build_model

class Profile(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nets, _ = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            setattr(self, name + '_ema', module)
        self.to(self.device)
   
    def parameter_count(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()    
        return num_params / 1e6
            
    @torch.no_grad()
    def evaluate(self):
        print_networks = False
        args = self.args
        nets = self.nets
        
        input_image = torch.randn(1, 3, args.img_size, args.img_size).to(self.device)
        y_index = torch.zeros(1,dtype = torch.long).to(self.device)
        laten_noise = torch.randn(1, args.latent_dim).to(self.device)
        style_vector = torch.randn(1, args.style_dim).to(self.device)
        
        style_encoder_params = self.parameter_count(nets.style_encoder)
        mapping_network_params = self.parameter_count(nets.mapping_network)
        generator_params = self.parameter_count(nets.generator)
        discriminator_params = self.parameter_count(nets.discriminator)

        if print_networks:
            print('discriminator')
            print(nets.discriminator)
        print('[discriminator] Total number of parameters : %.3f M' % (discriminator_params))
        discriminator_macs = profile_macs(nets.discriminator, (input_image, y_index))
        print('[discriminator] Total number of MACs : %.3f GMACs' % (discriminator_macs/1e9))
        
        if print_networks:
            print('style_encoder')
            print(nets.style_encoder)
        print('[style_encoder] Total number of parameters : %.3f M' % (style_encoder_params))
        style_encoder_macs = profile_macs(nets.style_encoder, (input_image, y_index))
        print('[style_encoder] Total number of MACs : %.3f GMACs' % (style_encoder_macs/1e9))
        
        if print_networks:
            print('mapping_network')
            print(nets.mapping_network)
        print('[mapping_network] Total number of parameters : %.3f M' % (mapping_network_params))
        mapping_network_macs = profile_macs(nets.mapping_network, (laten_noise, y_index))
        print('[mapping_network] Total number of MACs : %.3f MegaMACs' % (mapping_network_macs/1e6))
        
        if print_networks:
            print('generator')
            print(nets.generator)
        print('[generator] Total number of parameters : %.3f M' % (generator_params))
        generator_macs = profile_macs(nets.generator, (input_image, style_vector))
        print('[generator] Total number of MACs : %.3f GMACs' % (generator_macs/1e9))
        
        print('Total Parameters : %.3f M' % (generator_params + mapping_network_params + style_encoder_params))
        print('Total number of MACs : %.3f GMACs' % ((generator_macs + mapping_network_macs + style_encoder_macs)/1e9))