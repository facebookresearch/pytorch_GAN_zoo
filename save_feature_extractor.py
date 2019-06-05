import torch
import argparse
import torchvision
from models.loss_criterions.loss_texture import LossTexture
from models.networks.constant_net import MeanStd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        help="""Name of the desured featire extractor:
                        - vgg19, vgg16 : a variation of the style transfer \
                        feature developped in \
                        http://arxiv.org/abs/1703.06868""")
    parser.add_argument('--layers', type=int, nargs='*', default=None)
    parser.add_argument('output_path', type=str,
                        help="""Path of the output feature extractor""")

    args = parser.parse_args()

    if args.model_name in ["vgg19", "vgg16"]:
        if args.layers is None:
            args.layers = [3, 4, 5]
        featureExtractor = LossTexture(torch.device("cpu"),
                                       args.model_name,
                                       args.layers)
        featureExtractor.saveModel(args.output_path)
    elif args.model_name == "inception":
        model = torchvision.models.inception_v3(pretrained=True)
        model.AuxLogits.fc = MeanStd()
        model.fc = MeanStd()
        mean = [2*p - 1 for p in[0.485, 0.456, 0.406]]
        std = [2*p for p in [0.229, 0.224, 0.225]]
        torch.save(dict(model=model, fullDump=True,
                        mean=mean, std=std), args.output_path)
    else:
        raise AttributeError(args.model_name + " not implemented yet")
