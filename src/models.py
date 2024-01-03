import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sklearn import tree
import xgboost as xgb

from base_models import NeuralNetwork, ParallelNetworks
import pdb 
import numpy as np

def build_model(conf):
    if conf.family == "gpt2":
        if ('use_first_n_layer' not in conf) or conf.use_first_n_layer is None:
            use_first_n_layer = conf.n_layer
        else:
            use_first_n_layer=conf.use_first_n_layer
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            use_first_n_layer=use_first_n_layer
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, \
                 output_attentions=False, use_first_n_layer=12):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)
        if use_first_n_layer > n_layer:
            self.first_n_layer = n_layer
        else:
            self.first_n_layer = use_first_n_layer
        self.output_attentions = output_attentions
        if self.first_n_layer < n_layer:
            self.output_attentions = True
        
        print("use first n layer:", self.first_n_layer)
        
        

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds,output_attentions=self.output_attentions, output_hidden_states=True)
        if self.output_attentions:
            prediction = self._read_out(output.hidden_states[self.first_n_layer])
        else:
            prediction = self._read_out(output.last_hidden_state)
        if not self.output_attentions:
            return prediction[:, ::2, 0][:, inds]           # predict only on xs
        else:
            return prediction[:, ::2, 0][:, inds], output  


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None, output_w=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        if output_w:
            w_prediction = []
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                if output_w:
                    w_prediction.append(torch.zeros_like(xs[:, 0]))
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            if output_w:
                w_prediction.append(ws[:,:,0])
            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        if output_w:
            return torch.stack(preds, dim=1), torch.stack(w_prediction).permute(1,0,2)
        else:
            return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModelGradientDescent:
    def __init__(self, n_steps=3, step_size=1.0, weight_decay=0.0):
        self.n_steps = n_steps
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.name = f"OLS_GD_steps={n_steps}"

    def gradient_descent(self, X, y):
        # w = torch.rand(X.shape[1], 1) / np.sqrt(X.shape[1])
        w = torch.zeros(X.shape[1], 1)
        
        for _ in range(self.n_steps):
            grad = (X.T @ X) @ w - (X.T@y)[:, None]
            updates = self.step_size * grad  + self.weight_decay * w
            w = w - updates

        return w

    def __call__(self, xs, ys, inds=None, return_all_ws=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)

            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]
                
                w = self.gradient_descent(train_x, train_y)
                ws[b] = w

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] = ws
        if return_all_ws:
            return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)

class LeastSquaresModelOnlineGradientDescent:
    def __init__(self, step_size=1.0, weight_decay=0.0):
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.name = f"OLS_OGD_step_size={step_size}"

    def gradient_descent(self, X, y):
        w = torch.rand(X.shape[1], 1)
        n_samples = X.shape[0]
        for i in range(n_samples):
            xi = X[i][:,None]; yi=y[i]
            grad = xi @ xi.T @ w - yi * xi
            updates = self.step_size * grad  + self.weight_decay * w
            w = w - updates

        return w

    def __call__(self, xs, ys, inds=None, return_all_ws=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)

            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]
                
                w = self.gradient_descent(train_x, train_y)
                ws[b] = w

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] = ws
        if return_all_ws:
            return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)
        
class LeastSquaresModelAdaptiveOnlineGradientDescent:
    def __init__(self):
        self.name = f"OLS_Adaptive_OGD"

    def gradient_descent(self, X, y):
        w = torch.rand(X.shape[1], 1) / np.sqrt(X.shape[1])
        n_samples = X.shape[0]
        for i in range(n_samples):
            xi = X[i][:,None]; yi=y[i]
            grad = xi @ xi.T @ w - yi * xi
            step_size = (1 / (xi.T @ xi)).item()
            updates = step_size * grad
            w = w - updates

        return w

    def __call__(self, xs, ys, inds=None, return_all_ws=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)

            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]
                
                w = self.gradient_descent(train_x, train_y)
                ws[b] = w

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] = ws
        if return_all_ws:
            return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)

class LeastSquaresModelNewtonMethod:
    def __init__(self, n_newton_steps=3):
        self.n_newton_steps = n_newton_steps
        self.name = f"OLS_newton_inv={n_newton_steps}"

    def newton_inv(self, A):
        lam = torch.linalg.norm(A @ A.T)
        alpha = 2/lam
        inv = alpha * A.T 
        eye = torch.eye(A.shape[0])
        
        for i in range(self.n_newton_steps):
            inv = inv @ (2*eye-A@inv)

        return inv

    def __call__(self, xs, ys, inds=None, return_all_ws=False, return_all_invs=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        all_invs = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            invs = []
            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)
            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]
                
                X = train_x.T @ train_x
                y = train_x.T @ train_y
                inv = self.newton_inv(X)
                invs.append(inv[None,])
                w = inv @ y[:,None] 
                
                ws[b] = w
            
            invs = torch.cat(invs, dim=0)
            all_invs[i] = invs
                    

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] =  ws

        if return_all_ws:
            if return_all_invs:
                return torch.stack(preds, dim=1), all_ws, all_invs
            else:
                return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)


class KernelRegression:
    def __init__(self, kernel="linear", alpha=1.0, degree=2):
        self.name = f"kernel_regression"
        self.kernel = kernel 
        self.alpha = alpha 
        self.degree = degree #degree for polynomial kernel


    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)
            pred = []
            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]
                krr = KernelRidge(alpha=self.alpha, kernel=self.kernel, degree=self.degree)
                krr.fit(train_x, train_y)
                pred_y = krr.predict(test_x[b])
                pred.append(pred_y)
            pred = torch.FloatTensor(np.array(pred)).flatten() 
            preds.append(pred)
            

        return torch.stack(preds, dim=1)

class ProjectionModel:
    def __init__(self):
        self.name = f"projection"


    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)
            pred = []
            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]
                x_q = test_x[b]

                pred_y = torch.tensor(0.0)
                for i, x in enumerate(train_x):
                    x_normalized = x/torch.linalg.norm(x, ord=2)
                    proj = torch.inner(x_q.flatten()/torch.linalg.norm(x_q, ord=2), x_normalized.flatten())
                    pred_y += proj * train_y[i]
                    
                pred.append(pred_y)
            pred = torch.FloatTensor(np.array(pred)).flatten() 
            preds.append(pred)
            

        return torch.stack(preds, dim=1)

class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Ridge regression (for regularized linear regression).
# Seems to take more time as we decrease alpha.
class RidgeModel:
    def __init__(self, alpha, max_iter=100000):
        # the l2 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"ridge_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Ridge(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)
    
# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
