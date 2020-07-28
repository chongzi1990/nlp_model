# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
import preprocessing.process_cl_data as pp
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='text_classification', help='task type: text_classification, sequence_labeling')
    parser.add_argument("--model", type=str, default='lr', help='task type')
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--test", type=int, default=1, help="0: Don't infer, 1: Infer")
    parser.add_argument("--split_data", type=int, default=0, help="0: Don't split, 1: split data")
    args = parser.parse_args()
    data_path = "./data/"+args.task+"/"
    if args.task == "text_classification":
        data_processor = pp.ClPreprocessor(data_path, ".txt")
        if args.split_data == 1:
            data_processor.split_file("waimai_10k.csv", ".txt")
        data_processor.get_train_data()
        '''
        if args.model == "lr":
            if args.train == 1:
                train_and_fit(args)
            if args.test == 1:
                inferer = infer_from_trained(args, detect_entities=True)
        '''