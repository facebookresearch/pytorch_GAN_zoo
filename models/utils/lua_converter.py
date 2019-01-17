import torch.legacy.nn as lnn
import torch.nn as nn

class LinearView(nn.Module):

    def __init__(self, size):

        super(LinearView, self).__init__()
        self.size = size

    def forward(self, x):

        return x.view(-1, self.size)

def convertLegacyModule(module):

    if isinstance(module, lnn.SpatialConvolution):

        output = nn.Conv2d(module.nInputPlane, module.nOutputPlane,
                          (module.kH, module.kW),
                          stride = module.stride,
                          padding = (module.padH, module.padW))

        output.weight.data = module.weight.view(module.nOutputPlane, module.nInputPlane, module.kH, module.kW).clone()
        output.bias.data = module.bias.clone()

        return output

    if isinstance(module, lnn.SpatialBatchNormalization):

        output =  nn.BatchNorm2d(module.weight.size(0),
                                eps = module.eps,
                                momentum = module.momentum,
                                affine = module.affine)


        output.weight.data = module.weight.clone()
        output.bias.data = module.bias.clone()

        return output

    if isinstance(module, lnn.BatchNormalization):

        output =  nn.BatchNorm1d(module.weight.size(0),
                                eps = module.eps,
                                momentum = module.momentum,
                                affine = module.affine)

        output.weight.data = module.weight.clone()
        output.bias.data = module.bias.clone()

        return output

    if isinstance(module, lnn.ReLU):

        return nn.ReLU(inplace = module.inplace)

    if isinstance(module, lnn.SpatialMaxPooling):

        output = nn.MaxPool2d((module.kH, module.kW),
                              stride = (module.dH, module.dW),
                              padding = (module.padH, module.padW))

        return output

    if isinstance(module, lnn.Sequential):

        module_list = []

        for item in module.modules:
            module_list.append(convertLegacyModule(item))
        return nn.Sequential(*module_list)

    if isinstance(module, lnn.Linear):

        output = nn.Linear(module.weight.size(1), module.weight.size(0))
        output.weight.data = module.weight.clone()
        output.bias.data = module.bias.clone()

        return output

    if isinstance(module, lnn.View):

        return LinearView(module.size[0])

    if isinstance(module, lnn.Dropout):

        return nn.Dropout(p = module.p, inplace = module.inplace)

    if isinstance(module, lnn.Sigmoid):

        return nn.Sigmoid()

    raise AttributeError("Invalid module type " + str(type(module)))
