import sys
from yaml import dump
import numpy as np
import torch
import tensorflow as tf
from collections import OrderedDict
from os import path, mkdir, fsync
from tensorboard_logger import configure, log_value
import shutil
from time import time
import argparse
from copy import deepcopy
import os
import yaml
import json
import collections


def get_parser(arg_list=['default']):
    parser = argparse.ArgumentParser()
    if 'default' in arg_list:
        parser.add_argument("--seed", default=1234, help="", type=int)
        parser.add_argument("--experiment_name", default="experiment_name", help="")
        parser.add_argument("--code_name", default="code_name", help="")
        parser.add_argument("--cfg_filename", default="default", help="")
        parser.add_argument("--restore", default=False, help="whether to load from checkpoint", type=str2bool)
        parser.add_argument("--device", default="0", help="")
        parser.add_argument("--save_logs", default=False, help="whether to save sys output", type=str2bool)

    return parser

class Config(object):
    def __init__(self, args):
        # SET UP PATHS
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '../..'))
        self.paths['workspace'] = path.abspath(path.join(path.dirname(__file__), '..'))


        # Do Hyper-parameter sweep, if needed
        #self.idx = args.base + args.inc

        # Make results reproducible
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        tf.random.set_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Frequency of saving results and models.
        #self.save_after = args.max_episodes // args.save_count if args.max_episodes >= args.save_count else args.max_episodes

        # add path to models
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['Experiments'],args.experiment_name)
        self.paths["tb"] = path.join(self.paths['experiment'],'tb', args.code_name+"_"+str(args.seed)+"/")
        #self.paths["data"] = path.join(self.paths['Experiments'], args.env, 'data', args.experiment)
        #path_prefix = [self.paths['experiment'], str(args.seed)]
        self.paths['logs'] = path.join(self.paths['experiment'], 'Logs',args.code_name+"_"+str(args.seed)+"")
        self.paths['ckpt'] = path.join(self.paths['experiment'], 'Checkpoints/',args.code_name+"_"+str(args.seed)+"/")
        self.paths['results'] = path.join(self.paths['experiment'], 'Results/',args.code_name+"_"+str(args.seed)+"/")
        #self.paths['data'] = args.data_path
        self.paths['data'] = path.join(self.paths['experiment']  , 'Data/',args.code_name+"_"+str(args.seed)+"/")




        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets']:
                create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['logs'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        
        # Output logging
        if args.save_logs:
            sys.stdout = Logger(self.paths['logs'], args.restore)



        print("=====Configurations=====\n", args)

    def add_dict(self, dict):
        # Copy all the variables from args to config
        self.__dict__.update(dict)
        # Save the all the configuration settings
        dump(self.__dict__, open(path.join(self.paths['logs'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)


    def add(self,key_name, val):
        tmp = {}
        tmp[key_name] = val
        self.__dict__.update(tmp)
        # Save the all the configuration settings
        dump(self.__dict__, open(path.join(self.paths['logs'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)
    
    
    def load_config(self, subfolder, config_name, format='yaml'):
        with open(os.path.join(self.paths['workspace'], "cfgs", subfolder, "{}.{}".format(config_name, format)), "r") as f:
            if format == "json":
                config_dict = json.load(f)
            elif format =="yaml":
                config_dict = yaml.safe_load(f)
        config_dict = dict_update(config_dict, self.__dict__, recursive=False)
        self.add_dict(config_dict)


def dict_update(d, u, recursive=False):
    if recursive:
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
    else:
        for k, v in u.items():
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def str2bool(text):
    if text == 'True':
        arg = True
    elif text == 'False':
        arg = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    return arg




def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
                shutil.rmtree(dir_path)
                mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

"""
def create_directory_tree(dir_path):
    dir_path = str.split(dir_path, sep='/')[1:-1]  # Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join('/', *(dir_path[:i + 1])))
"""

def create_directory_tree(dir_path):
    dir_path = dir_path.strip(os.path.sep)  # Remove leading and trailing path separators
    dir_path_parts = dir_path.split(os.path.sep)

    # Handle Windows drive letter
    if os.name == 'nt' and len(dir_path_parts[0]) == 2 and dir_path_parts[0][1] == ':':
        start_index = 1  # Skip the drive letter part
    else:
        start_index = 0

    for i in range(start_index, len(dir_path_parts)):
        partial_path = os.path.sep.join(dir_path_parts[:i + 1])
        if not path.exists(partial_path):
            os.makedirs(partial_path)

def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore):
        self.terminal = sys.stdout
        #self.file = 'file' in method
        #self.term = 'term' in method
        self.file = True
        self.term = True

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        self.log.flush()
        fsync(self.log.fileno())

class Exp_Logger(object):
    def __init__(self, directory_name):
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def log_stat(self, key, value, t):
        if self.use_tb:
            self.tb_logger(key, value, t)




if __name__ == '__main__':
    pass