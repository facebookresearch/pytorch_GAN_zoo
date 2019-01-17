import visdom
import torch
import torchvision.transforms as Transforms
import torchvision.utils as vutils
import numpy as np
import random

vis = visdom.Visdom()

def resizeTensor(data, out_size_image):

    out_data_size = (data.size()[0], data.size()[1], out_size_image[0], out_size_image[1])

    outdata = torch.empty(out_data_size)
    data = torch.clamp(data, min=-1, max=1)

    interpolationMode = 0
    if out_size_image[0] < data.size()[0] and out_size_image[1] < data.size()[1]:
        interpolationMode = 2

    transform = Transforms.Compose([Transforms.Normalize((-1., -1., -1.), (2, 2, 2)),
                                    Transforms.ToPILImage(),
                                    Transforms.Resize(out_size_image, interpolation = interpolationMode),
                                    Transforms.ToTensor()])

    for img in range(out_data_size[0]):
        outdata[img] = transform(data[img])

    return outdata

def publishTensors(data, out_size_image, caption = "", window_token = None, env = "main", nrow= 16):
    global vis
    outdata = resizeTensor(data, out_size_image)
    return vis.images(outdata, opts=dict(caption=caption), win = window_token, env = env, nrow = nrow)

def saveTensor(data, out_size_image, path):
    outdata = resizeTensor(data, out_size_image)
    vutils.save_image(outdata, path)

def publishLoss(data, name = "", window_token = None, env="main"):

    nItems = len(data["G"])
    inputY = np.array([[data["G"][x], data["D"][x]] for x in range(nItems)])

    #inputY = np.array(data["G"])
    inputX = np.array(data["iter"])
    opts={'title': name + (' scale %d loss over time' % data["scale"]),
        'legend': ["G", "D"], 'xlabel': 'iteration','ylabel': 'loss'}

    return vis.line(X= inputX, Y= inputY, opts = opts, win=window_token, env = env)

def publishLinePlot(data, xData, name = "", window_token = None, env = "main"):

    if not isinstance(data, list):
        data = [data]

    inputX = np.array(xData)
    nItems = len(xData)
    inputY = np.array([[y[1][x] for y in data] for x in range(nItems)])

    opts={'title': name,
         'legend': [y[0] for y in data],
         'xlabel': 'iteration'}

    return vis.line(X= inputX, Y= inputY, opts = opts, win=window_token, env = env)

def delete_env(name):

    vis.delete_env(name)

def publishScatterPlot(data, name = "", window_token = None):
    r"""
    Draws 2D or 3d scatter plots

    Args:

        data (list of tensors): list of Ni x 2 or Ni x 3 tensors. Each tensor
                        representing a cloud of data
        name (string): plot name
        window_token (token): ID of the window the plot should be done

    Returns:

        ID of the window where the data are plotted
    """

    if not isinstance(data, list):
        raise ValueError("Input data should be a list of tensors")

    nPlots = len(data)
    colors = []

    random.seed(None)

    for item in range(nPlots):
        N = data[item].size()[0]
        colors.append(torch.randint(0, 256, (1, 3)).expand(N, 3))

    colors = torch.cat(colors, dim = 0).numpy()
    opts = {'markercolor': colors,
            'caption': name}
    activeData = torch.cat(data, dim = 0)

    return vis.scatter(activeData, opts = opts, win = window_token, name = name)
