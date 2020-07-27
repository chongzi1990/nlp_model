# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='classification', help='task type')
    parser.add_argument("--model", type=str, default='lr', help='task type')
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--test", type=int, default=1, help="0: Don't infer, 1: Infer")

    args = parser.parse_args()
    if args.task == "classification":
        if args.model == "lr":
            if args.train == 1:
                train_and_fit(args)

            if args.test == 1:
                inferer = infer_from_trained(args, detect_entities=True)
