import torch
import argparse
from models.loss_criterions.loss_texture import LossTexture

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        choices=["vgg19", "vgg16"],
                        help="""Name of the desured featire extractor:
                        - vgg19, vgg16 : a variation of the style transfer \
                        feature developped in \
                        http://arxiv.org/abs/1703.06868""")
    parser.add_argument('--layers', type=int, nargs='*',
                        help="For vgg models only. Layers to select. \
                        Default ones are 3, 4, 5.", default=None)
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
    else:
        raise AttributeError(args.model_name + " not implemented yet")
