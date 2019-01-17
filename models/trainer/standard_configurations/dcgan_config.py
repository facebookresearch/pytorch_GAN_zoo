from ..utils.config import BaseConfig

# Default configuration for ProgressiveGANTrainer
_C = BaseConfig()

############################################################

# Depth of a convolutional layer for each scale
_C.depth = 3

# Mini batch size
_C.miniBatchSize = 16

# Dimension of the latent vector
_C.dimLatentVector = 100

# Dimension of the output image
_C.dimOutput = 3

# Dimension of the generator
_C.dimG = 64

# Dimension of the discrimator
_C.dimD = 64

# We are doing an alternative training. Number of consecutive updates of G
_C.kInnerG = 1

# We are doing an alternative training. Number of consecutive updates of D
_C.kInnerD = 1

# Loss mode
_C.lossMode = 'DCGAN'

# Gradient penalty coefficient (WGANGP)
_C.lambdaGP = 0.

# Noise standard deviation in case of instance noise (0 <=> no Instance noise)
_C.sigmaNoise = 0.

# Weight penalty on |D(x)|^2
_C.epsilonD = 0.

# Base learning rate
_C.baseLearningRate = 0.0002

# Numbert of iterations between two updates
_C.batchAccumulation = 1

# In case of AC GAN, weight on the classification loss (per scale)
_C.weightConditionG = 0.0
_C.weightConditionD = 0.0

############################################
# In case of a product gan
############################################

# Set to True to load a product gan
_C.productGan = False

# Latent dimension of the noise vector given to the shape generator
_C.dimLatentVectorShape = 64

# Latent dimension of the noise vector given to the texture generator
_C.dimLatentVectorTexture = 64

# Inner dimension of the shape generator
_C.dimGShape = 64

# Inner dimension of the texture generator
_C.dimGTexture = 64

# If ACGAN, specifies which conditions should be transimitted to the shape
# generator, and which ones should be given to the texture generator
_C.keySplits = None
