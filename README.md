# Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models

This is an official repository for our paper, [Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models](https://arxiv.org/abs/2310.17086).


```bibtex
@misc{fu2023transformers,
    title={Transformers Learn Higher-Order Optimization Methods for In-Context Learning: A Study with Linear Models}, 
    author={Deqing Fu and Tian-Qi Chen and Robin Jia and Vatsal Sharan},
    year={2023},
    eprint={2310.17086},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Codes are mostly modified from [this prior work](https://github.com/dtsip/in-context-learning/).

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate transformers_icl_opt
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

3. Run probing for each Transformers layer

    ```
    cd src
    python probing.py
    ```

4. Compute Transformer's similarities to both Iterative Newton's Method and Gradient Descent
   
   ```
   python eval_similarity.py
   ```

   This will plot Fig. 1(a) and Fig. 3 in the [paper](https://arxiv.org/abs/2310.17086), under a new folder `eval`.