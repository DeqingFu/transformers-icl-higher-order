import json 
import os 
import subprocess
import numpy as np 
import shutil
from munch import Munch
import yaml
import copy
import time

n_layer = 12
n_head = 8

src = f"../models/linear_regression/pretrained"
config_path = f"{src}/config.yaml"
with open(config_path, "r") as fp:  # we don't Quinfig it to avoid inherits
    base_conf = Munch.fromDict(yaml.safe_load(fp))

for i in range(1, n_layer+1):
    dst = os.path.abspath(f"../models/by_layer_linear/probing_{i}")
    if not os.path.exists(dst):
        shutil.copytree(src, dst) 

for i in range(1, n_layer+1):
    probe_i_conf = copy.deepcopy(base_conf)

    probe_i_conf['model']['use_first_n_layer'] = i
    probe_i_conf['training']['resume_id'] = f'probing_{i}'
    probe_i_conf['training']['data'] = 'gaussian'
    probe_i_conf['wandb']['name'] = f"linear_regression_probing_{i}"
    probe_i_conf['wandb']['entity'] = '<your-wandb-username>'
    probe_i_conf['wandb']['project'] = 'transformers_icl_opt'
    probe_i_conf['out_dir'] = os.path.abspath(f"../models/by_layer_linear/")
     ## Remove curriculum
    probe_i_conf['training']['curriculum']['dims']['start'] = probe_i_conf['training']['curriculum']['dims']['end']
    probe_i_conf['training']['curriculum']['points']['start'] = probe_i_conf['training']['curriculum']['points']['end']

    probe_i_conf_path = os.path.abspath(f"./conf/linear_regression_probing_{i}.yaml")

   
    with open(probe_i_conf_path, "w") as fp:  # we don't Quinfig it to avoid inherits
        yaml.safe_dump(probe_i_conf, fp)

if not os.path.exists("logs"):
    os.mkdir("logs")

for i in range(1, n_layer+1):
    with open("probe.sh", "w") as fp:
        data = f"""#!/bin/bash
#SBATCH -o ./logs/slurm-%j.out # STDOUT
source ~/.bashrc
source activate transformers_icl_opt

"""

        data += f"python train_linear.py --config conf/linear_regression_probing_{i}.yaml"

        fp.write(data)
        os.chmod("probe.sh", 0o0777)
    
    subprocess.Popen("sbatch --gpus=1 --time=6:00:00 --cpus-per-task=8 probe.sh", shell=True, stdout=subprocess.PIPE)
    print(f"submitted for probing {i}")
    time.sleep(10)