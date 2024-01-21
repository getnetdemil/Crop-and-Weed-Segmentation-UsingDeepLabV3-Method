import torch
import torch.nn as nn
from torchvision.models import resnet
import torch.nn.functional as F




class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        
        # Use a pre-trained ResNet backbone
        self.backbone = resnet.resnet101(pretrained=True)
        
        # Modify the backbone for semantic segmentation
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Define the ASPP module
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # Define the decoder module
        self.decoder = Decoder(low_level_channels=256, output_channels=256)
        
        # Define the final prediction layer
        # self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)
        # self.final_conv = nn.Conv2d(5, num_classes, kernel_size=1)
        
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Backbone forward pass
        x = self.backbone(x)
        low_level_features = x   # Extract features from an intermediate layer (e.g., layer1)
        print(f"Low-level features size: {low_level_features.size()}")

        # ASPP module forward pass
        aspp_output = self.aspp(x)
        print(f"ASPP output size: {aspp_output.size()}")
        
        # Decoder module forward pass
        decoder_output = self.decoder(low_level_features, aspp_output)
        print(f"Decoder output size: {decoder_output.size()}")
        
        # Final prediction layer forward pass
        x = self.final_conv(decoder_output)
        print(f"Final prediction output size: {x.size()}")
                
        return x

class ASPP(nn.Module):
    def __init__ (self, in_channels, out_channels):
        super(ASPP, self).__init__()
                
        #ASPP modelue implementation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.image_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels + 4 * out_channels, out_channels, kernel_size=1) #added in_channels + 4 * out_channels
    def forward(self, x):
        
        # ASPP module forward pass implementation
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        out4 = F.relu(self.conv4(x))
        out_pooling = self.image_pooling(x)
        
        # Reshape out_pooling to match spatial dimensions of other tensors
        out_pooling = out_pooling.expand(x.size(0), -1, x.size(2), x.size(3)) 
        
        print(f"layer 1 output {out1.size()}")
        print(f"layer 2 output {out2.size()}")
        print(f"layer 3 output {out3.size()}")
        print(f"layer 4 output {out4.size()}")
        print(f"pooling layer output {out_pooling.size()}")
         
        # concatnate the output along the channel dimension
        concatnated = torch.cat((out1, out2, out3, out4, out_pooling), dim=1)
        print(f"concatenated layer output {concatnated.size()}")
        
        out = F.relu(self.conv5(concatnated))
        print(f"output layer {out.size()}")
        
        return out

class Decoder(nn.Module):
    def __init__(self, low_level_channels, output_channels):
        super(Decoder, self).__init__()
        
        #Defining the layers of the Decoder Module
        # low level feature processing
        self.conv1 = nn.Conv2d(low_level_channels, 48, kernel_size=1)
        # self.conv1 = nn.Conv2d(low_level_channels, 256, kernel_size=1)
        
        # concatnation and further processing to make feature map dimensions compatible
        self.conv2 = nn.Conv2d(2096, output_channels, kernel_size=3)
        
        #upsampling to get the final prediction map
        self.upsample = nn.Upsample(scale_factor=4)
    def forward(self, x, low_level_features):
        # Implementing the forward pass for the Decoder Module
        
        #upsampling ASPP output by 4
        x = self.upsample(x)
        print(f"Upsampled aspp output size: {x.size()}")      
 
        # low_level_features=self.upsample(low_level_features)
        print(f"low_level_features size before conv{low_level_features.size()}")
        # processing low level feature
        low_level_features = self.conv1(low_level_features)
        print(f"low_level_features size after the 1st conv1{low_level_features.size()}")        
        
        # Use interpolate to upsample width and height dimensions
        low_level_features = F.interpolate(low_level_features, scale_factor=(4, 4), mode='nearest')
        print(f"dimension Upsampled output size: {low_level_features.size()}")
        
        # concatenating the ASPP and low level features 
        x = torch.cat((x, low_level_features), dim=1)
        print(f"concatenated at decoder output size: {x.size()}")
        
        x=self.conv2(x)
        print(f"after conv 2 output size: {x.size()}")
        
        #further processing after concatination
        x = self.upsample(x)
        print(f"final upsampled output size: {x.size()}")
                
        return x
    