from super_image import EdsrModel, EdsrConfig
import torch
import torch.nn as nn

class EDSRModel(EdsrModel):
    def __init__(self, in_channels,feature_channels, out_channels, scaling_factor):
        
        config = EdsrConfig(scale=scaling_factor)
        
        super(EDSRModel, self).__init__(config)
        
        self._modify_architecture(in_channels, feature_channels, out_channels)
        
        def _modify_architecture(self, in_channels, feature_channels, out_channels):
            
            del self.sub_mean
            del self.add_mean
            
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and m.in_channels == 64 and m.out_channels == 64:
                    m.in_channels = feature_channels
                    m.out_channels = feature_channels
                    # he initialization
                    m.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(feature_channels, feature_channels, m.kernel_size[0], m.kernel_size[1])))
                    m.bias = nn.Parameter(torch.zeros(feature_channels))
                    
            self.head = nn.Conv2d(in_channels, feature_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1))

            self.tail = self._flexible_upscaler(in_channels, feature_channels, out_channels, scaling_factor)
            
        def _flexible_upscaler(self, in_channels, feature_channels, out_channels, scaling_factor):
            layers = []
            num_channels = int(4 * feature_channels)
            
            layers.append(nn.Conv2d(feature_channels, num_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
            layers.append(nn.PixelShuffle(upscale_factor = 2))
            
            for _ in range(scaling_factor / 4):
                layers.extend([
                    nn.Conv2d(num_channels, num_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                    nn.PixelShuffle(upscale_factor = 2)
                ])
                
            layers.append(nn.Conv2d(num_channels, in_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
            
            return nn.Sequential(*layers)
                          