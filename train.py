import os
import importlib
import argparse
import visualization.visualizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        help='Name of the model to launch, available models are\
                         PGAN and PPGAN. To get all possible option for a model\
                          please run train.py $MODEL_NAME -overrides')
    parser.add_argument('--no_vis', help='Print more data',
                        action='store_true')
    parser.add_argument('--np_vis', help=' Replace visdom by a numpy based visualizer (SLURM)',
                        action='store_true')
    parser.add_argument('--restart', help=' If a checkpoint is detected, do not try to load it',
                        action='store_true')
    parser.add_argument('-n','--name', help="Model's name",
                        type=str, dest="name")
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir", default='output_networks')
    parser.add_argument('-c','--config', help="Model's name",
                        type=str, dest="configPath")
    parser.add_argument('-s', '--save_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved. In the case of a\
                        evaluation test, iteration to work on.",
                        type=int, dest="saveIter")
    parser.add_argument('-e', '--eval_iter', help="If it applies, frequence at\
                        which a checkpoint should be saved",
                        type=int, dest="evalIter")
    parser.add_argument('-S', '--Scale_iter', help="If it applies, scale to work\
                        on")
    parser.add_argument('-v','--partitionValue', help="Partition's value",
                        type=str, dest="partition_value")
    parser.add_argument('-A','--statsFile', help="Statistsics file",
                        type=str, dest="statsFile")

    baseArgs, unknown = parser.parse_known_args()

    vis_module = None
    if baseArgs.np_vis:
        vis_module = importlib.import_module("visualization.np_visualizer")
    elif baseArgs.no_vis:
        print("Visualization disabled")
    else:
        vis_module = importlib.import_module("visualization.visualizer")

    module = importlib.import_module( "models.train." + baseArgs.model_name)

    if not os.path.isdir(baseArgs.dir):
        os.mkdir(baseArgs.dir)

    print("Running " + baseArgs.model_name)
    output = module.train(parser, visualization = vis_module)

    if output is not None and not output:
        print("...FAIL")

    else:
        print("...OK")
