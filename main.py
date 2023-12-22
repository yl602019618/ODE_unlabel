import hydra
import utils
import torch
import logging
from train import ODE_Identifier
logger = logging.getLogger(__name__)
import shutil
import os


import numpy as np

@hydra.main(config_path="config", config_name="config_odelinear2d", version_base="1.3")
def main(cfg):
    shutil.rmtree('results')
    os.mkdir('results')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    utils.set_seed_everywhere(cfg.seed)
    cfgs = cfg.train
    if cfgs.dim == 2:
        x_init =  np.array([2.0,0.0])
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'
    integrator_keywords['atol'] = 1e-12
    model = ODE_Identifier(cfgs,x_init)
    model.train(cfgs.epoch)






if __name__ == "__main__":
    main()
