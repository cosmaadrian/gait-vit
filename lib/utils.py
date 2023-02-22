import os
import glob

import torch
import numpy as np

def load_model(args):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/../checkpoints/{args.group}:{args.name}/{args.checkpoint_kind}/*.ckpt'
    print("::: Loading model from", checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)

    if len(checkpoints) != 1:
        best_checkpoints = [sorted(checkpoints, key = lambda x: float(x.split('=')[-1][:-5]))[-1]]
        # HACK
        print(":::::::::::")
        print(":::::::::::")
        print("::::::::::: WARNING ::::::::::")
        print("::::::::::: Found multiple checkpoints:")
        for c in checkpoints:
            print(c)
        print("::::::::::: Using the best one: ", best_checkpoints[0])
        print("::::::::::: WARNING ::::::::::")
        print(":::::::::::")
        print(":::::::::::")

        checkpoints = best_checkpoints

    assert len(checkpoints) == 1

    try:
        state_dict = torch.load(checkpoints[-1])
    except Exception as e:
        print("No checkpoints found: ", checkpoint_path)
        raise e

    return state_dict

def load_model_by_dir(name):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/{name}/*.ckpt'
    print('::: Loading model from:', checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)

    assert len(checkpoints) == 1

    return torch.load(checkpoints[-1])

def load_model_by_name(name, kind = 'best'):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/../checkpoints/{name}/{kind}/*.ckpt'
    print('::: Loading model from:', checkpoint_path)

    checkpoints = glob.glob(checkpoint_path)

    assert len(checkpoints) == 1

    return torch.load(checkpoints[-1])
