from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns
import torch
from tqdm import tqdm
import json 

from samplers import get_data_sampler
from tasks import get_task_sampler
import models
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from eval import get_run_metrics, read_run_dir, get_model_from_run
from sklearn.metrics.pairwise import cosine_similarity
import pdb 
from matplotlib.colors import LogNorm

palette = sns.color_palette('colorblind')
plt.rcParams["font.family"] = "DejaVu Serif"
run_dir = "../models"

task = "linear_regression"
run_id = "pretrained"
run_path = os.path.join(run_dir, task, run_id)

model, conf = get_model_from_run(run_path)
model.eval()
model.output_attentions=True

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size * 5

data_sampler = get_data_sampler('gaussian', n_dims)
task_sampler = get_task_sampler(
    conf.training.task,
    n_dims,
    batch_size
)

task = task_sampler()
xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end)
ys = task.evaluate(xs)

with torch.no_grad():
    ys_pred, output = model(xs, ys)

suffix = "_standard"
model_base_path = f"../models/by_layer_linear{suffix}"
n_layers = conf.model['n_layer']
metric = task.get_metric()
n_data = 41


eval_folder = f"../eval"
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)

folder = f"{eval_folder}/resid_sim_{run_id}/"
if not os.path.exists(folder):
    os.makedirs(folder)



tf_layer_index = list(range(1, n_layers+1))

def get_tf_residual(layer_index):
    model_i, _ = get_model_from_run(f"{model_base_path}/probing_{layer_index}")
    readout = model_i._read_out
    hidden_state = output.hidden_states[i]
    tf_pred = readout(hidden_state)[:,::2,0]
    tf_resid = (tf_pred - ys).detach().numpy()
    return tf_resid
    


def get_average_resid_similarity(tf_resid, method_resid, batch_size=batch_size):
    return np.mean([cosine_similarity(
                        tf_resid[k, :].reshape(1, -1), 
                        method_resid[k,:].reshape(1, -1)
                    ).flatten()[0] for k in range(batch_size)])

def plot_correlation(
        tf_layer_to_resid,
        target_model_name='newton', 
        possible_hidden_layer=tf_layer_index, 
        possible_model_params=range(1, 25),
        **kwargs):
    
    figsize = (16, 20)
    print("figure size:", figsize)
    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    target_model_to_resid = {}
    for param in tqdm(possible_model_params):
        if target_model_name == 'newton':
            lstsq_model = models.LeastSquaresModelNewtonMethod(n_newton_steps=param)
        elif target_model_name == 'gd':
            lr = 0.01 if 'lr' not in kwargs else kwargs['lr']
            lstsq_model = models.LeastSquaresModelGradientDescent(n_steps=param, step_size=lr)
        elif target_model_name == 'knn':
            lstsq_model = models.NNModel(n_neighbors=param)
        elif target_model_name == 'ogd':
            lstsq_model = models.LeastSquaresModelOnlineGradientDescent(step_size=param)
        method_pred = lstsq_model(xs, ys)
        method_resid = (method_pred - ys).detach().numpy()
        assert not np.isnan(method_resid).any()
        if np.isnan(method_resid).any():
            pdb.set_trace()
        target_model_to_resid[param] = method_resid
    

    heatmap = np.zeros((len(possible_hidden_layer), len(possible_model_params)))

    for i, n_layer in enumerate(possible_hidden_layer):
        for j, param in enumerate(possible_model_params):
            tf_resid = tf_layer_to_resid[n_layer]
            model_resid = target_model_to_resid[param]
            sim = get_average_resid_similarity(tf_resid, model_resid)
            heatmap[i,j] = sim 

    try:
        if 'suffix' in kwargs:
            with open(f"{folder}/resid_sim_heatmap_tf_{target_model_name}_{kwargs['suffix']}.json", "w") as fp:
                json.dump(heatmap.tolist(), fp)
        else:
            with open(f"{folder}/resid_sim_heatmap_tf_{target_model_name}.json", "w") as fp:
                json.dump(heatmap.tolist(), fp)
    except Exception as e:
        print("Failed to cache with error", e)

    s = sns.heatmap(heatmap.T, annot=True, cmap='Spectral_r', \
                xticklabels=possible_hidden_layer, yticklabels=possible_model_params,
                ax=ax, fmt='.4f', vmin=0.4, vmax=1.0,
                cbar_kws={"orientation": "horizontal", "pad":0.02},
                annot_kws={"size": 12})
    s.set(xlabel='Layer Index', ylabel='Iterative Steps')
    
    ret_dict = {}
    for col, _ in enumerate(heatmap):
        row = np.argmax(heatmap[col])
        ax.add_patch(Rectangle((col, row),1,1, fill=False, edgecolor='red', lw=4))
        ret_dict[tf_layer_index[col]] = possible_model_params[row]
    
    if target_model_name == 'newton':
        ax.set_title(f"Similarity of Residuals (Transformers v.s. Iterative Newton)", fontsize=16)
    if target_model_name == 'gd':
        ax.set_title(f"Similarity of Residuals (Transformers v.s. Gradient Descent)", fontsize=16)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='y', rotation=0)

    if 'suffix' in kwargs:
        fig.savefig(f"{folder}/resid_sim_heatmap_tf_{target_model_name}_{kwargs['suffix']}.pdf", 
                bbox_inches='tight', pad_inches=0, dpi=400)
    else:
        fig.savefig(f"{folder}/resid_sim_heatmap_tf_{target_model_name}.pdf", 
                    bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close(fig)
    return ret_dict


tf_layer_to_resid = {}
for i in tqdm(tf_layer_index):
    tf_layer_to_resid[i] = get_tf_residual(i)

lr = 0.01
transformers_newton_map = plot_correlation(tf_layer_to_resid, target_model_name='newton', 
                                           possible_model_params=range(1,24))
print("newton\n", transformers_newton_map)


gd_steps = [1] + list(range(0, 1310, 50))[1:]
print(gd_steps)
transformers_gd_map = plot_correlation(tf_layer_to_resid, target_model_name='gd',
                                       possible_model_params=gd_steps, 
                                       lr=lr)
print("gd\n", transformers_gd_map)

## Similarity with GD in log scale
gd_steps = [int(2**exponent) for exponent in range(0, 13)]
print(gd_steps)
transformers_gd_map_log = plot_correlation(tf_layer_to_resid, target_model_name='gd',
                                       possible_model_params=gd_steps, 
                                       lr=lr, suffix='log')
print("gd (log)\n", transformers_gd_map_log)

with torch.no_grad():
    sim_tf_newton_over_layer = []
    sim_tf_gd_over_layer = []
    sim_tf_min_norm_over_layer = []
    for i in tqdm(tf_layer_index):
        tf_resid = tf_layer_to_resid[i]

        newton_model = models.LeastSquaresModelNewtonMethod(n_newton_steps=transformers_newton_map[i])
        newton_resid = (newton_model(xs,ys) - ys).detach().numpy()

        gd_model = models.LeastSquaresModelGradientDescent(n_steps=transformers_gd_map[i],step_size=lr)
        gd_resid = (gd_model(xs,ys) - ys).detach().numpy()

        min_norm_model = models.LeastSquaresModel()
        min_norm_resid = (min_norm_model(xs,ys) - ys).detach().numpy()

        
        sim_tf_newton_over_layer.append(get_average_resid_similarity(tf_resid, newton_resid))
        sim_tf_gd_over_layer.append(get_average_resid_similarity(tf_resid, gd_resid))
        sim_tf_min_norm_over_layer.append(get_average_resid_similarity(tf_resid, min_norm_resid))
        
        
        
    

    plt.plot(tf_layer_index, sim_tf_newton_over_layer, color='red', label='SimE(Transformers, Newton)')
    plt.plot(tf_layer_index, sim_tf_gd_over_layer, color='blue', label='SimE(Transformers, GD)')
    plt.plot(tf_layer_index, sim_tf_min_norm_over_layer, color='green', label='SimE(Transformers, OLS)')
    plt.legend()
    plt.xlabel("Layer Index", fontsize=16)
    plt.ylabel("Cosine Similarity", fontsize=16)
    plt.savefig(folder + f"over_layers.pdf", \
                bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.clf()
