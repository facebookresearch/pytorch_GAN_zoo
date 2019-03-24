import os
import json
from nevergrad.optimization import optimizerlib
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..gan_visualizer import GANVisualizer
from ..utils.utils import loadmodule, getLastCheckPoint, getVal, \
    getNameAndPackage
from ..utils.image_transform import standardTransform
from ..metrics.nn_score import buildFeatureExtractor
from ..networks.constant_net import FeatureTransform

def pil_loader(path):

    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def getFeatireSize(x):

    s = x.size()
    out=1
    for p in s[1:]:
        out*=p

    return out


class IDModule(nn.Module):

    def __init__(self):

        super(IDModule, self).__init__()
        # self.dummy = nn.Conv2d(1,1,1,1)

    def forward(self, x):
        return x.view(-1, getFeatireSize(x))


def updateParser(parser):

    parser.add_argument('-f', '--featureExtractor', help="Path to the feature \
                        extractor", nargs='*',
                        type=str, dest="featureExtractor")
    parser.add_argument('--input_images', type=str, dest="inputImage",
                        help="Path to the input image.")
    parser.add_argument('-N', type=int, dest="nRuns",
                        help="Number of gradient descent to run at the same \
                        time. Being too greedy may result in memory error.",
                        default=1)
    parser.add_argument('-l', type=float, dest="learningRate",
                        help="Learning rate",
                        default=1)
    parser.add_argument('-S', '--suffix', type=str, dest='suffix',
                        help="Output's suffix", default="inspiration")
    parser.add_argument('-R', '--rLoss', type=float, dest='lambdaD',
                        help="Realism penalty", default=0.03)
    parser.add_argument('--nSteps', type=int, dest='nSteps',
                        help="Number of steps", default=6000)
    parser.add_argument('--weights', type=float, dest='weights',
                        nargs='*', help="Weight of each classifier. Default \
                        value is one. If specified, the number of weights must\
                        match the number of feature exatrcators.")
    parser.add_argument('--gradient_descent', help='gradient descent',
                        action='store_true')
    parser.add_argument('--lbfgs', help='lbfgs',
                        action='store_true')
    parser.add_argument('--unfixed', help='unfixed',
                        action='store_true')
    parser.add_argument('--random_search', help='Random search',
                        action='store_true')
    parser.add_argument('--nevergradcma', help='CMA nevergrad',
                        action='store_true')
    parser.add_argument('--nevergradde', help='DE nevergrad',
                        action='store_true')
    parser.add_argument('--nevergradpso', help='PSO nevergrad',
                        action='store_true')
    parser.add_argument('--nevergrad2pde', help='2PDE nevergrad',
                        action='store_true')
    parser.add_argument('--nevergradopo', help='2PDE nevergrad',
                        action='store_true')
    parser.add_argument('--nevergraddopo', help='2PDE nevergrad',
                        action='store_true')
    parser.add_argument('--nevergradpdopo', help='2PDE nevergrad',
                        action='store_true')
    parser.add_argument('--save_descent', help='Save descent',
                        action='store_true')

    return parser


def gradientDescentOnInput(model,
                           input,
                           featureExtractors,
                           imageTransforms,
                           weights=None,
                           visualizer=None,
                           lambdaD=0.03,
                           nSteps=6000,
                           randomSearch=False,
                           lbfgs=False,
                           unfixed=False,
                           nevergradcma=False,
                           nevergradde=False,
                           nevergradpso=False,
                           nevergrad2pde=False,
                           nevergradopo=False,
                           nevergraddopo=False,
                           nevergradpdopo=False,
                           lr=1,
                           outPathSave=None):
    r"""
    Performs a similarity search with gradient descent.

    Args:

        model (BaseGAN): trained GAN model to use
        input (tensor): inspiration images for the gradient descent. It should
                        be a [NxCxWxH] tensor with N the number of image, C the
                        number of color channels (typically 3), W the image
                        width and H the image height
        featureExtractors (nn.module): list of networks used to extract features
                                       from an image
        weights (list of float): if not None, weight to give to each feature
                                 extractor in the loss criterion
        visualizer (visualizer): if not None, visualizer to use to plot
                                 intermediate results
        lambdaD (float): weight of the realism loss
        nSteps (int): number of steps to perform
        randomSearch (bool): if true, replace tha gradient descent by a random
                             search
        nevergradcma (bool): if true, replace tha gradient descent by a 
                             CMA-ES
        nevergradpso (bool): if true, replace tha gradient descent by a 
                             PSO
        nevergradde (bool): if true, replace tha gradient descent by a 
                             DE
        nevergrad2pde (bool): if true, replace tha gradient descent by a 
                             2points-DE
        nevergradopo (bool): if true, replace tha gradient descent by a 
                             one-plus-one
        nevergraddopo (bool): if true, replace tha gradient descent by a 
                             discrete one-plus-one
        nevergradpdopo (bool): if true, replace tha gradient descent by a 
                             portfolio discrete one-plus-one
        lr (float): learning rate of the gradient descent
        outPathSave (string): if not None, path to save the intermediate iterations
                              of the gradient descent

    Returns

        output, optimalVector, optimalLoss

        output (tensor): output images
        optimalVector (tensor): latent vectors corresponding to the output
                                images
    """
    lbfgs = unfixed or lbfgs
    nevergrad = nevergradcma or nevergradde or nevergradpso or nevergrad2pde or nevergradopo or nevergraddopo or nevergradpdopo
    randomSearch = randomSearch or nevergrad
    print("Running for %d setps" % nSteps)

    if visualizer is not None:
        visualizer.publishTensors(input, (128, 128))

    # Detect categories
    varNoise = torch.randn((input.size(0),
                            model.config.noiseVectorDim +
                            model.config.categoryVectorDim),
                           requires_grad=True, device=model.device)

    if lbfgs:
        optimNoise = optim.LBFGS([varNoise],
                            lr=lr)
    else:
        optimNoise = optim.Adam([varNoise],
                            betas=[0., 0.99], lr=lr)

    noiseOut = model.test(varNoise, getAvG=True, toCPU=False)

    if not isinstance(featureExtractors, list):
        featureExtractors = [featureExtractors]
    if not isinstance(imageTransforms, list):
        imageTransforms = [imageTransforms]

    nExtractors = len(featureExtractors)

    if weights is None:
        weights = [1.0 for i in range(nExtractors)]

    if len(imageTransforms) != nExtractors:
        raise ValueError(
            "The number of image transforms should match the number of \
            feature extractors")
    if len(weights) != nExtractors:
        raise ValueError(
            "The number of weights should match the number of feature\
             extractors")

    featuresIn = []

    for i in range(nExtractors):

        if len(featureExtractors[i]._modules) > 0:
            featureExtractors[i] = nn.DataParallel(
                featureExtractors[i]).train().to(model.device)

        imageTransforms[i] = nn.DataParallel(
            imageTransforms[i]).to(model.device)

        featuresIn.append(featureExtractors[i](
            imageTransforms[i](input.to(model.device))).detach())

    lr = 1

    optimalVector = None
    optimalLoss = None

    epochStep = int(nSteps / 3)
    gradientDecay = 0.1

    nImages = input.size(0)
    if randomSearch and nevergrad:
        optimizer_name = "CMA" if nevergradcma else "DE" if nevergradde else "PSO" if nevergradpso else "TwoPointsDE" if nevergrad2pde else "PortfolioDiscreteOnePlusOne" if nevergradpdopo else "DiscreteOnePlusOne" if nevergraddopo else "OnePlusOne" if nevergradopo else "ERROR"
        optimizers = []
        for i in range(nImages):
            optimizers += [optimizerlib.registry[optimizer_name](dimension=model.config.noiseVectorDim+model.config.categoryVectorDim,budget=nSteps)]
    def resetVar(newVal):
        newVal.requires_grad = True
        print("Updating the optimizer with learning rate : %f" % lr)
        varNoise = newVal
        optimNoise = optim.Adam([varNoise],
                                betas=[0., 0.99], lr=lr)

    # String's format for loss output
    formatCommand = ' '.join(['{:>4}' for x in range(nImages)]) 
    backtobfgs = False
    for iter in range(nSteps):

        optimNoise.zero_grad()
        model.netG.zero_grad()
        model.netD.zero_grad()
        if randomSearch:
            varNoise = torch.randn((nImages,
                                    model.config.noiseVectorDim +
                                    model.config.categoryVectorDim),
                                   requires_grad=True, device=model.device)
            if nevergrad:
                inps = []
                for i in range(nImages):
                    inps += [optimizers[i].ask()]
                    npinps = np.array(inps)

                varNoise = torch.tensor(npinps, dtype=torch.float32, device=torch.device('cuda:0'))#.astype(np.float32)
                #varNoise = torch.from_numpy(npinps).to(torch.cuda.FloatTensor)#.astype(np.float32)
                varNoise.requires_grad = True
                varNoise.to(model.device)

        noiseOut = model.avgG(varNoise)
        sumLoss = torch.zeros(nImages, device=model.device)

        loss = (((varNoise**2).mean(dim=1) - 1)**2)
        if os.environ["renorm"] == "renorm":
            sumLoss += loss.view(nImages)
        loss.sum(dim=0).backward(retain_graph=True)

        for i in range(nExtractors):
            featureOut = featureExtractors[i](imageTransforms[i](noiseOut))
            diff = ((featuresIn[i] - featureOut)**2)
            loss = weights[i] * diff.mean(dim=1)
            sumLoss += loss

            if not randomSearch:
                retainGraph = (lambdaD > 0) or (i != nExtractors - 1)
                loss.sum(dim=0).backward(retain_graph=retainGraph)

        if lambdaD > 0:

            loss = -lambdaD * model.netD(noiseOut)[:, 0]
            sumLoss += loss

            if not randomSearch:
                loss.sum(dim=0).backward()

        if nevergrad:
            for i in range(nImages):
                 optimizers[i].tell(inps[i], float(sumLoss[i]))
                    
                 
        if not randomSearch:
#            def closure():
#                optimizer.zero_grad()
#                out = seq(input)
#                loss = criterion(out, target)
#                loss.backward()
#                return loss

            optimNoise.step(closure=lambda: loss.sum(dim=0))

        if optimalLoss is None:
            optimalVector = deepcopy(varNoise)
            optimalLoss = sumLoss
        else:
            optimalVector = torch.where(sumLoss.view(-1, 1) < optimalLoss.view(-1, 1),
                                        varNoise, optimalVector).detach()
            optimalLoss = torch.where(sumLoss < optimalLoss,
                                      sumLoss, optimalLoss).detach()
            if lbfgs: 
                for i in range(nImages):
                    if sumLoss[i] != sumLoss[i] and not unfixed:
                     #   varNoise[i].data = optimalVector[i].data
                        varNoise.data = optimalVector.data
                        print("yoyo")
                        optimNoise = optim.Adam([varNoise],
                            lr=lr)
                        backtobfgs = True
            if backtobfgs:
                optimNoise = optim.LBFGS([varNoise], lr=lr)
                backtobfgs = False

        if iter % 100 == 0:
            if visualizer is not None:
                visualizer.publishTensors(noiseOut.cpu(), (128, 128))

                if outPathSave is not None:
                    index_str = str(int(iter/100))
                    outPath = os.path.join(outPathSave, index_str + ".jpg")
                    visualizer.saveTensor(
                        noiseOut.cpu(),
                        (noiseOut.size(2), noiseOut.size(3)),
                        outPath)

            print(str(iter) + " : " + formatCommand.format(
                *["{:10.6f}".format(sumLoss[i].item())
                  for i in range(nImages)]))

        if iter % epochStep == (epochStep - 1):
            lr *= gradientDecay
            resetVar(optimalVector)

    print(optimalVector.size(), varNoise.size())
    output = model.test(optimalVector, getAvG=True, toCPU=True).detach()

    if visualizer is not None:
        visualizer.publishTensors(
            output.cpu(), (output.size(2), output.size(3)))

    print("optimal losses : " + formatCommand.format(
        *["{:10.6f}".format(optimalLoss[i].item())
          for i in range(nImages)]))
    return output, optimalVector, optimalLoss


def test(parser, visualisation=None):

    parser = updateParser(parser)

    kwargs = vars(parser.parse_args())

    # Parameters
    name = getVal(kwargs, "name", None)
    if name is None:
        raise ValueError("You need to input a name")

    module = getVal(kwargs, "module", None)
    if module is None:
        raise ValueError("You need to input a module")

    imgPath = getVal(kwargs, "inputImage", None)
    if imgPath is None:
        raise ValueError("You need to input an image path")

    scale = getVal(kwargs, "scale", None)
    iter = getVal(kwargs, "iter", None)
    nRuns = getVal(kwargs, "nRuns", 1)

    checkPointDir = os.path.join(kwargs["dir"], name)
    print("toto")
    checkpointData = getLastCheckPoint(checkPointDir,
                                       name,
                                       scale=scale,
                                       iter=iter)
    weights = getVal(kwargs, 'weights', None)

    if checkpointData is None:
        raise FileNotFoundError(
            "Not checkpoint found for model " + str(name) + " at directory " + str(checkPointDir) + 'cwd=' + str( os.getcwd()))

    modelConfig, pathModel, _ = checkpointData

    keysLabels = None
    with open(modelConfig, 'rb') as file:
        keysLabels = json.load(file)["attribKeysOrder"]
    if keysLabels is None:
        keysLabels = {}

    packageStr, modelTypeStr = getNameAndPackage(module)
    modelType = loadmodule(packageStr, modelTypeStr)

    visualizer = GANVisualizer(
        pathModel, modelConfig, modelType, visualisation)

    # Load the image
    targetSize = visualizer.model.getSize()

    baseTransform = standardTransform(targetSize)

    img = pil_loader(imgPath)
    input = baseTransform(img)
    input = input.view(1, input.size(0), input.size(1), input.size(2))

    pathsModel = getVal(kwargs, "featureExtractor", None)
    featureExtractors = []
    imgTransforms = []

    if weights is not None:
        if pathsModel is None or len(pathsModel) != len(weights):
            raise AttributeError(
                "The number of weights must match the number of models")

    if pathsModel is not None:
        for path in pathsModel:
            if path == "id":
                featureExtractor = IDModule()
                imgTransform = IDModule()
            else:
                featureExtractor, mean, std = buildFeatureExtractor(
                    path, resetGrad=True)
                imgTransform = FeatureTransform(mean, std, size=128)  # None)
            featureExtractors.append(featureExtractor)
            imgTransforms.append(imgTransform)
    else:
        featureExtractors = IDModule()
        imgTransforms = IDModule()

    outVectorList = []
    basePath = os.path.splitext(imgPath)[0] + "_" + kwargs['suffix']

    if not os.path.isdir(basePath):
        os.mkdir(basePath)

    basePath = os.path.join(basePath, os.path.basename(basePath))

    print("All results will be saved in " + basePath)

    outDictData = {}
    outPathDescent = None

    fullInputs = torch.cat([input for x in range(nRuns)], dim=0)

    if kwargs['save_descent']:
        outPathDescent = os.path.join(
            os.path.dirname(basePath), "descent")
        if not os.path.isdir(outPathDescent):
            os.mkdir(outPathDescent)

    img, outVectors, loss = gradientDescentOnInput(visualizer.model,
                                                   fullInputs,
                                                   featureExtractors,
                                                   imgTransforms,
                                                   visualizer=visualisation,
                                                   lambdaD=kwargs['lambdaD'],
                                                   nSteps=kwargs['nSteps'],
                                                   weights=weights,
                                                   randomSearch=kwargs['random_search'],
                                                   lbfgs=kwargs['lbfgs'],
                                                   unfixed=kwargs['unfixed'],
                                                   nevergradcma=kwargs['nevergradcma'],
                                                   nevergradpso=kwargs['nevergradpso'],
                                                   nevergradde=kwargs['nevergradde'],
                                                   nevergrad2pde=kwargs['nevergrad2pde'],
                                                   nevergradopo=kwargs['nevergradopo'],
                                                   nevergraddopo=kwargs['nevergraddopo'],
                                                   nevergradpdopo=kwargs['nevergradpdopo'],
                                                   lr=kwargs['learningRate'],
                                                   outPathSave=outPathDescent)
    
    
    pathVectors = basePath + "vector.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))

    path = basePath + ".jpg"
    visualisation.saveTensor(img, (img.size(2), img.size(3)), path)
    outDictData[os.path.splitext(os.path.basename(path))[0]] = \
        [x.item() for x in loss]

    outVectors = outVectors.view(outVectors.size(0), -1)
    outVectors *= torch.rsqrt((outVectors**2).mean(dim=1, keepdim=True))

    barycenter = outVectors.mean(dim=0)
    barycenter *= torch.rsqrt((barycenter**2).mean())
    meanAngles = (outVectors * barycenter).mean(dim=1)
    meanDist = torch.sqrt(((barycenter-outVectors)**2).mean(dim=1)).mean(dim=0)
    outDictData["Barycenter"] = {"meanDist": meanDist.item(),
                                 "stdAngles": meanAngles.std().item(),
                                 "meanAngles": meanAngles.mean().item()}

    path = basePath + "_data.json"
    outDictData["kwargs"] = kwargs

    with open(path, 'w') as file:
        json.dump(outDictData, file, indent=2)

    pathVectors = basePath + "vectors.pt"
    torch.save(outVectors, open(pathVectors, 'wb'))
