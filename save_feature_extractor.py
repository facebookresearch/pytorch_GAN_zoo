import torch
import argparse

from models.loss_criterions.loss_texture import LossTexture

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        help="""Name of the desured featire extractor:
                        - st : a variation of the style transfer feature developped in http://arxiv.org/abs/1703.06868""")
    parser.add_argument('output_path', type=str,
                        help="""Path of the output feature extractor""")

    args = parser.parse_args()

    if args.model_name == 'st':
        featureExtractor = LossTexture(torch.device("cpu"), "vgg19", [3, 4, 5])
        featureExtractor.saveModel(args.output_path)

    else:
        raise AttributeError(args.model_name + " not implemented yet")
