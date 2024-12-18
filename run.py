from utils.utils import *
from exp import *

if __name__ == '__main__':
    args = arg_parser()
    exp = Exp(args)
    if args.aggregator == "fedprox":
        exp.train(f"{args.data}_{args.model_name}_{args.aggregator}_{args.mu}")
    else:
        exp.train(f"{args.data}_{args.model_name}_{args.aggregator}")