from ...utils.config import BaseConfig

# Default configuration for ProgressiveGANTrainer
_C = BaseConfig()

# TODO : build something epoch based

# Maximum number of iteration at each scale
_C.maxIterAtScale = [48000, 96000, 96000, 96000, 96000, 96000, 96000, 96000, 200000]

# Blending mode.

############################################################

# 2 possible values are possible:
# - custom: iterations at which alpha should be updated and new value after the update
# are fully described by the user
# - linear: The user just inputs the number of updates of alpha, and the number of iterations
# between two updates for each scale
_C.alphaJumpMode = "linear"

# If _C.alphaJumpMode == "custom", then the following fields should be completed

# For each scale, iteration at wich the blending factor alpha should be
# updated
_C.iterAlphaJump = [[], [0, 1000, 2000], [0, 1000, 4000, 8000, 16000],
                   [0, 2000, 4000, 8000]]

# New value of the blending factor alpha during the update (see above)
_C.alphaJumpVals = [ [], [1., 0.5, 0], [1, 0.75, 0.5, 0.25, 0.], [1., 0.75, 0.5, 0.]]

# If _C.alphaJumpMode == "linear", then the following fields should be completed

# Number of jumps per scale
_C.alphaNJumps = [0, 600, 600, 600, 600, 600, 1200, 1200, 1200]

# Number of iterations between two jumps
_C.alphaSizeJumps = [0, 32, 32, 32, 32, 32, 32, 32, 32, 32]

#############################################################

# Mini batch size
_C.miniBatchSize = 16

# Dimension of the latent vector
_C.dimLatentVector = 512

# Latent dimension of the noise vector given to the shape generator
_C.latentTexture = 256

# Latent dimension of the noise vector given to the shape generator
_C.latentShape = 256

# Set to true if the shape and the texture generator should share the same noise
# data
_C.mixedNoise = False

# We are doing an alternative training. Number of consecutive updates of G
_C.kInnerG = 1

# We are doing an alternative training. Number of consecutive updates of D
_C.kInnerD = 1

# Should bias be initialized to zero ?
_C.initBiasToZero = True

# Per channel normalization
_C.perChannelNormalization = True

# Loss mode
_C.lossMode = 'WGANGP'

# Gradient penalty coefficient (WGANGP)
_C.lambdaGP = 10.

# Leakyness of the leakyRelU activation function
_C.leakyness = 0.2

# Weight penalty on |D(x)|^2
_C.epsilonD = 0.001

# Mini batch regularization
_C.miniBatchStdDev = True

# Base learning rate
_C.baseLearningRate = 0.001

# Numbert of iterations between two updates
_C.batchAccumulation = 1

# RGB or grey level output ?
_C.dimOutput = 3

# In case of AC GAN, weight on the classification loss (per scale)
_C.weightConditionG = 1.0
_C.weightConditionD = 1.0

#Equalized learning rate
_C.equalizedlR = True

# If ACGAN, specifies which conditions should be transimitted to the shape
# generator, and which ones should be given to the texture generator
_C.keySplits = None

# Order of the keys. Generated automatically, don't specofy this value in your
# configuration
_C.attribKeysOrder = None

# Depth of a convolutional layer for each scale
_C.depthScales = [[512, 256, 256], [512, 256, 256], [512,256, 256],
  [512, 256, 256], [256, 256, 256], [128, 128, 128], [64, 64, 64], [32, 32, 32]]

# Do we change the AC-GAN classification weights in order to equalize the
# different class of the input dataset ?
_C.equalizeLabels = False

# Do we have a shape discriminator in our model ? Don't specify this value in
# your configuration
_C.maskExtraction = False

# Texture loss (cf style gan)
_C.textureLossModel = ""
_C.textureLossLayers = [1, 2, 3, 4]
_C.weightTextureLoss = 0.
