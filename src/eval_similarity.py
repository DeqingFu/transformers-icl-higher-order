from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import matplotlib
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

model_base_path = f"../models/by_layer_linear"
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

cmap = matplotlib.cm.get_cmap('PuRd')
def get_color(cmap, idx, max_idx):
    return cmap(0.2 + (idx+1)/(max_idx+1))

# MSE over examples 
plt.figure(figsize=(16,9))
with torch.no_grad():
     for i in tqdm(range(1, n_layers+1)):
        if i not in tf_layer_index:
            continue 
        model_i, _ = get_model_from_run(f"{model_base_path}/probing_{i}")
        readout = model_i._read_out
        hidden_state = output.hidden_states[i]
        pred_at_i = readout(hidden_state)[:,::2,0]
        loss = metric(pred_at_i, ys).detach().numpy() / 20
        plt.plot(loss.mean(axis=0), lw=1, label=f"Hidden Layer #{i}", \
                 color=get_color(cmap, i, max(tf_layer_index)))


plt.yscale("log")
plt.legend(fontsize=12, loc='lower left')
plt.title("Average MSE Loss v.s. # In-Context Examples")
plt.savefig(eval_folder + "/mse_over_examples.pdf", bbox_inches='tight', pad_inches=0.0, dpi=400)
plt.clf()

def get_tf_residual(layer_index):
    print(f"Getting TF residual for layer {layer_index}")
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
            print("Error: NaN in residuals")
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