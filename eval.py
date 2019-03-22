import importlib
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script', add_help=False)
    parser.add_argument('evaluation_name', type=str,
                        help='Name of the evaluation method to launch. To get \
                        the arguments specific to an evaluation method please \
                        use: eval.py evaluation_name -h')
    parser.add_argument('--no_vis', help='Print more data',
                        action='store_true')
    parser.add_argument('--np_vis', help=' Replace visdom by a numpy based \
                        visualizer (SLURM)',
                        action='store_true')
    parser.add_argument('-m', '--module', help="Module to evaluate, available\
                        modules: PGAN, PPGAN, DCGAN",
                        type=str, dest="module")
    parser.add_argument('-n', '--name', help="Model's name",
                        type=str, dest="name")
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir", default="output_networks")
    parser.add_argument('-i', '--iter', help='Iteration to evaluate',
                        type=int, dest="iter")
    parser.add_argument('-s', '--scale', help='Scale to evaluate',
                        type=int, dest="scale")
    parser.add_argument('-c', '--config', help='Training configuration',
                        type=str, dest="config")
    parser.add_argument('-v', '--partitionValue', help="Partition's value",
                        type=str, dest="partition_value")
    parser.add_argument("-A", "--statsFile", dest="statsFile",
                        type=str, help="Path to the statistics file")

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()

    vis_module = None
    if args.np_vis:
        vis_module = importlib.import_module("visualization.np_visualizer")
    elif args.no_vis:
        print("Visualization disabled")
    else:
        vis_module = importlib.import_module("visualization.visualizer")

    module = importlib.import_module("models.eval." + args.evaluation_name)
    print("Running " + args.evaluation_name)

    parser.add_argument('-h', '--help', action='help')
    out = module.test(parser, visualisation=vis_module)

    if out is not None and not out:
        print("...FAIL")

    else:
        print("...OK")
