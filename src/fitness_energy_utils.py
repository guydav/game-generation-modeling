from collections import defaultdict
import copy
from dataclasses import dataclass
from difflib import HtmlDiff
from itertools import zip_longest
import typing

from IPython.display import display, Markdown, HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from statsmodels.distributions.empirical_distribution import ECDF
from tabulate import tabulate
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, IterableDataset

from fitness_features_preprocessing import NON_FEATURE_COLUMNS



FITNESS_DATA_FILE = '../data/fitness_scores.csv.gz'



def _find_nth(text, target, n):
    start = text.find(target)
    while start >= 0 and n > 1:
        start = text.find(target, start+len(target))
        n -= 1
    return start


def load_fitness_data(path: str = FITNESS_DATA_FILE) -> pd.DataFrame:
    fitness_df = pd.read_csv(path)
    fitness_df = fitness_df.assign(real=fitness_df.src_file == 'interactive-beta.pddl', original_game_name=fitness_df.game_name)
    fitness_df.original_game_name.where(
        fitness_df.game_name.apply(lambda s: (s.count('-') <= 1) or (s.startswith('game-id') and s.count('-') >= 2)),
        fitness_df.original_game_name.apply(lambda s: s[:_find_nth(s, '-', 2)]),
        inplace=True)

    fitness_df.columns = [c.replace(' ', '_').replace('(:', '') for c in fitness_df.columns]
    fitness_df = fitness_df.assign(real=fitness_df.real.astype('int'))
    fitness_df = fitness_df[list(fitness_df.columns[:4]) + list(fitness_df.columns[-2:]) + list(fitness_df.columns[4:-2])]
    return fitness_df


DEFAULT_RANDOM_SEED = 33
DEFAULT_TRAINING_PROP = 0.8


def train_test_split_by_game_name(df: pd.DataFrame, training_prop: float = DEFAULT_TRAINING_PROP,
    random_seed: int = DEFAULT_RANDOM_SEED, positive_column: str = 'real', positive_value: typing.Any = True):

    real_game_names = df[df[positive_column] == positive_value].original_game_name.unique()

    train_game_names, test_game_names = train_test_split(real_game_names, train_size=training_prop, random_state=random_seed)
    train_df = df[df.game_name.isin(train_game_names) | df.original_game_name.isin(train_game_names)]
    test_df = df[df.game_name.isin(test_game_names) | df.original_game_name.isin(test_game_names)]
    return train_df, test_df


def df_to_tensor(df: pd.DataFrame, feature_columns: typing.List[str],
    positive_column: str = 'real', positive_value: typing.Any = True):
    return torch.tensor(
        np.stack([
            np.concatenate((
                df.loc[df.real & (df.original_game_name == game_name), feature_columns].to_numpy(),
                df.loc[(~df.real) & (df.original_game_name == game_name), feature_columns].to_numpy()
            ))
            for game_name
            in df[df[positive_column] == positive_value].original_game_name.unique()
        ]),
        dtype=torch.float
    )


def make_init_weight_function(bias: float = 0.01):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(bias)
    return init_weights


class CustomSklearnScaler:
    def __init__(self, passthrough: bool = False):
        self.passthrough = passthrough
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        if X.ndim != 3:
            raise ValueError('X must be 3D')

        if self.passthrough:
            return self

        self.mean = X.mean(axis=(0, 1))
        self.std = X.std(axis=(0, 1))
        self.std[torch.isclose(self.std, torch.zeros_like(self.std))] = 1
        return self

    def transform(self, X, y=None):
        if self.passthrough:
            return X
        return (X - self.mean) / self.std

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        return [f'x{i}' for i in range(self.mean.shape[0])]  # type: ignore

    def set_params(self, **params):
        if params:
            if 'passthrough' in params:
                self.passthrough = params['passthrough']

        return self

    def get_params(self, deep=True):
        return dict(passthrough=self.passthrough)


class FitnessEnergyModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: typing.Optional[int] = None,
        hidden_activation: typing.Callable = torch.relu,
        output_activation: typing.Optional[typing.Callable] = None,
        n_outputs: int = 1):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        if output_activation is None:
            output_activation = nn.Identity()
        self.output_activation = output_activation

        if hidden_size is None:
            self.fc1 = nn.Linear(self.n_features, self.n_outputs)
            self.hidden_activation = None

        else:
            self.fc1 = nn.Linear(self.n_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, self.n_outputs)
            self.hidden_activation = hidden_activation

    def forward(self, x, activate: bool = True):
        x = self.fc1(x)

        if self.hidden_activation is not None:
            x = self.hidden_activation(x)
            x = self.fc2(x)

        if self.n_outputs == 1 and activate and self.output_activation is not None:
            x = self.output_activation(x)

        return x


def _reduce(X: torch.Tensor, reduction: str, dim: typing.Optional[int] = None):
    if reduction == 'mean':
        if dim is None:
            return X.mean()
        return X.mean(dim=dim)
    elif reduction == 'sum':
        if dim is None:
            return X.sum()
        return X.sum(dim=dim)
    elif reduction.lower() == 'none':
        return X
    else:
        raise ValueError(f'Invalid reduction: {reduction}')


def fitness_nce_loss(scores: torch.Tensor, negative_score_reduction: str = 'sum', reduction: str = 'mean'):
    positive_scores = torch.log(scores[:, 0])
    negative_scores = _reduce(torch.log(1 - scores[:, 1:]), negative_score_reduction, dim=1)
    return _reduce(-(positive_scores + negative_scores), reduction)


def fitness_hinge_loss(scores: torch.Tensor, margin: float = 1.0, negative_score_reduction: str = 'none', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        positive_scores = positive_scores.unsqueeze(-1)
    return _reduce(torch.relu(positive_scores + margin - negative_scores), reduction)


def fitness_hinge_loss_with_cross_example(scores: torch.Tensor, margin: float = 1.0, alpha: float = 0.5,
    negative_score_reduction: str = 'none', reduction: str = 'mean'):
    hinge = fitness_hinge_loss(scores, margin, negative_score_reduction, reduction)

    positive_scores = scores[:, 0, None]
    negative_scores = scores[:, 1:]
    cross_example_loss = _reduce(torch.relu(positive_scores + margin - negative_scores), reduction)

    return alpha * hinge + (1 - alpha) * cross_example_loss


def fitness_log_loss(scores: torch.Tensor, negative_score_reduction: str = 'none', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    # negative_scores = scores[:, 1:].sum(dim=1)
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        positive_scores = positive_scores.unsqueeze(-1)
    return _reduce(torch.log(1 + torch.exp(positive_scores - negative_scores)), reduction)


def fitness_square_square_loss(scores: torch.Tensor, margin: float = 1.0, negative_score_reduction: str = 'none', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    # negative_scores = scores[:, 1:].sum(dim=1)
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        return fitness_square_square_loss_positive_negative_split(positive_scores, negative_scores, margin, reduction)

    return _reduce(positive_scores.pow(2) + torch.relu(margin - negative_scores).pow(2), reduction)


def fitness_square_square_loss_positive_negative_split(positive_scores: torch.Tensor, negative_scores: torch.Tensor, margin: float = 1.0, reduction: str = 'mean'):
    return _reduce(positive_scores.pow(2), reduction) + _reduce(torch.relu(margin - negative_scores).pow(2), reduction)


def fitness_softmin_loss(scores: torch.Tensor, beta: float = 1.0, reduction: str = 'mean'):
    return nn.functional.cross_entropy(
        - beta * scores,
        torch.zeros((scores.shape[0], 1), dtype=torch.long, device=scores.device),
        reduction=reduction)


def fitness_softmin_hybrid_loss(scores: torch.Tensor, margin: float = 1.0, beta: float = 1.0, reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    negative_scores = scores[:, 1:]
    negative_scores_softmin = nn.functional.softmin(beta * negative_scores, dim=1)
    effective_negative_scores = torch.einsum('bij, bjk -> bik', negative_scores.squeeze(-1).unsqueeze(1), negative_scores_softmin).squeeze(-1).squeeze()
    return _reduce(torch.relu(positive_scores + margin - effective_negative_scores), reduction)
    # return _reduce(torch.log(1 + torch.exp(positive_scores - negative_scores)) + torch.relu(margin - negative_scores), reduction)



DEFAULT_MODEL_PARAMS = {
    'n_features': None,
    'hidden_size': None,
    'hidden_activation': torch.relu,
    'n_outputs': 1,
    'output_activation': torch.sigmoid,
}

DEFAULT_TRAIN_KWARGS = {
    'weight_decay': 0.0,
    'lr': 1e-2,
    'loss_function': fitness_nce_loss,
    'should_print': False,
    'print_interval': 10,
    'patience_epochs': 5,
    'patience_threshold': 0.01,
    'batch_size': 8,
    'k': 4,
    'device': 'cpu',
    'random_seed': 33,
}

LOSS_FUNCTION_KAWRG_KEYS = ('margin', 'alpha', 'beta', 'negative_score_reduction', 'reduction')
DEFAULT_TRAIN_KWARGS.update({k: None for k in LOSS_FUNCTION_KAWRG_KEYS})


class SklearnFitnessWrapper:
    def __init__(self,
        model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        loss_function_kwarg_keys: typing.Sequence[str] = LOSS_FUNCTION_KAWRG_KEYS,
        **params):

        self.model_kwargs = copy.deepcopy(DEFAULT_MODEL_PARAMS)
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        self.train_kwargs = copy.deepcopy(DEFAULT_TRAIN_KWARGS)
        if train_kwargs is not None:
            self.train_kwargs.update(train_kwargs)

        self.loss_function_kwargs = {}
        self.loss_function_kwarg_keys = loss_function_kwarg_keys

        self.set_params(**params)

    def get_params(self, deep: bool = True) -> typing.Dict[str, typing.Any]:
        return {
            **self.model_kwargs,
            **self.train_kwargs,
        }

    def set_params(self, **params) -> 'SklearnFitnessWrapper':
        for key, value in params.items():
            if key in self.model_kwargs:
                self.model_kwargs[key] = value
            elif key in self.train_kwargs:
                self.train_kwargs[key] = value
            else:
                raise ValueError(f'Unknown parameter {key}')

        return self

    def fit(self, X, y=None) -> 'SklearnFitnessWrapper':
        self.model = FitnessEnergyModel(**self.model_kwargs)
        # if 'margin' in self.train_kwargs:
        #     init_weights = make_init_weight_function(self.train_kwargs['margin'] / 2)
        # else:
        init_weights = make_init_weight_function()
        self.model.apply(init_weights)
        train_kwarg_keys = list(self.train_kwargs.keys())
        for key in train_kwarg_keys:
            if key in self.loss_function_kwarg_keys:
                value = self.train_kwargs.pop(key)
                if value is not None:
                    self.loss_function_kwargs[key] = value

        self.train_kwargs['loss_function_kwargs'] = self.loss_function_kwargs
        self.model = train_and_validate_model(self.model, X, **self.train_kwargs)
        return self

    def transform(self, X, y=None) -> torch.Tensor:
        return self.model(X)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if self.model is not None:
            return self.model(*args, **kwargs)

        return torch.empty(0)



ModelClasses = typing.Union[nn.Module, SklearnFitnessWrapper, Pipeline]



def _score_samples(model: ModelClasses, X: torch.Tensor, y: typing.Optional[torch.Tensor],
    device: str = 'cpu') -> typing.Tuple[torch.Tensor, torch.Tensor]:

    with torch.no_grad():
        if isinstance(model, Pipeline):
            model.named_steps['fitness'].model.to(device)
            model.named_steps['fitness'].model.eval()
            scores = model.transform(X.to(device))

        elif isinstance(model, SklearnFitnessWrapper):
            model.model.to(device)
            model.model.eval()
            scores = model.transform(X.to(device))

        else:
            model.to(device)
            model.eval()
            scores = model(X.to(device), activate=False)

        scores = scores.detach().cpu()

    positive_scores = scores[:, 0]
    negative_scores = scores[:, 1:]
    return positive_scores.detach(), negative_scores.detach()


# def evaluate_fitness(model: ModelClasses, X: torch.Tensor, y: typing.Optional[torch.Tensor] = None,
#     score_sign: int = 1):
#     positive_scores, negative_scores = _score_samples(model, X, y)

#     game_average_scores = (positive_scores - negative_scores.mean(dim=1)) * score_sign
#     return game_average_scores.mean().item()


# def evaluate_fitness_flipped_sign(model: ModelClasses,
#     X: torch.Tensor, y=None):
#     return _score_samples(model, X, y, score_sign=-1)


def evaluate_fitness_overall_ecdf(model: ModelClasses,
    X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _score_samples(model, X, y)
    positive_scores = positive_scores.squeeze().cpu().numpy()
    negative_scores = negative_scores.squeeze().cpu().numpy()
    ecdf = ECDF(np.concatenate([positive_scores, negative_scores.reshape(-1)]))

    positive_mean_quantile = ecdf(positive_scores).mean()
    return -positive_mean_quantile


def evaluate_fitness_single_game_rank(model: ModelClasses, X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _score_samples(model, X, y)
    single_game_rank = (positive_scores[:, None] < negative_scores).float().mean(axis=1)  # type: ignore
    return single_game_rank.mean().item()


def build_multiple_scoring_function(
    evaluators: typing.Sequence[typing.Callable[[ModelClasses, torch.Tensor, typing.Optional[torch.Tensor]], float]],
    names: typing.Sequence[str]
    ) -> typing.Callable[[ModelClasses, torch.Tensor, typing.Optional[torch.Tensor]], typing.Dict[str, float]]:
    def _evaluate_fitness_multiple(model: ModelClasses, X: torch.Tensor, y=None, return_all=False):
        return {name: evaluator(model, X, y) for name, evaluator in zip(names, evaluators)}

    return _evaluate_fitness_multiple


@dataclass
class ConstrativeTrainingData:
    positive_samples: torch.Tensor
    negative_samples: torch.Tensor

    def __init_(self, positive_samples: torch.Tensor, negative_samples: typing.Union[torch.Tensor, typing.List[torch.Tensor]]):
        self.positive_samples = positive_samples
        if not isinstance(negative_samples, torch.Tensor):
            negative_samples = torch.cat(negative_samples, dim=0)
        self.negative_samples = negative_samples


DEFAULT_INITIAL_ENERGY = 5.0
DEFAULT_ENERGY_BETA = 1.0


class EnergyRecencyWeightedDataset(IterableDataset):
    current_epoch: int
    data: ConstrativeTrainingData
    energy_beta: float
    initial_energy: float
    k: int
    n_positives: int
    negative_energies: torch.Tensor
    negative_last_sampled: torch.Tensor
    positive_order: torch.Tensor

    def __init__(self, data: ConstrativeTrainingData, k: int = 4,
                 energy_beta: float = DEFAULT_ENERGY_BETA, initial_energy: float = DEFAULT_INITIAL_ENERGY):
        self.data = data
        self.k = k
        self.energy_beta = energy_beta
        self.initial_energy = initial_energy

        self.n_positives = len(data.positive_samples)
        self.negative_energies = torch.ones(len(data.negative_samples)) * initial_energy
        self.negative_last_sampled = torch.zeros(len(data.negative_samples))
        self.current_epoch = 0

        self.latest_negative_indices_yielded = []

    def _new_epoch(self):
        self.current_epoch += 1
        unnormalized_logprobs = self.current_epoch - self.negative_last_sampled - (self.negative_energies * self.energy_beta)
        shifted_logprobs = unnormalized_logprobs - unnormalized_logprobs.max()
        probs = torch.exp(shifted_logprobs)
        probs = probs / probs.sum()

        negative_indices = torch.multinomial(probs, self.n_positives * self.k, replacement=False)
        self.negative_last_sampled[negative_indices] = self.current_epoch
        self.negative_indices_per_positive = negative_indices.view(self.n_positives, self.k)

        self.positive_order = torch.randperm(self.n_positives)

    def __iter__(self):
        self._new_epoch()
        for positive_index in self.positive_order:
            negative_indices = self.negative_indices_per_positive[positive_index]
            self.latest_negative_indices_yielded.append(negative_indices)
            yield torch.cat((self.data.positive_samples[positive_index].unsqueeze(0), self.data.negative_samples[negative_indices]))

    def update_negative_energies(self, negative_energies: torch.Tensor):
        negative_indices = torch.stack(self.latest_negative_indices_yielded)
        if negative_indices.shape != negative_energies.shape:
            raise ValueError(f"Negative indices shape {negative_indices.shape} does not match negative energies shape {negative_energies.shape}")

        self.negative_energies[negative_indices.ravel()] = negative_energies.ravel()
        self.latest_negative_indices_yielded = []


def train_and_validate_model_weighted_sampling(
        model: nn.Module,
        train_data: ConstrativeTrainingData,
        val_data: typing.Optional[ConstrativeTrainingData] = None,
        loss_function: typing.Callable = fitness_square_square_loss,
        loss_function_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
        val_loss_function: typing.Callable = fitness_square_square_loss_positive_negative_split,
        optimizer_class: typing.Callable = torch.optim.SGD,
        n_epochs: int = 1000, lr: float = 0.01, weight_decay: float = 0.0,
        should_print: bool = True, should_print_weights: bool = False, print_interval: int = 10,
        patience_epochs: int = 5, patience_threshold: float = 0.01,
        batch_size: int = 8, k: int = 4,
        dataset_energy_beta: float = DEFAULT_ENERGY_BETA,
        dataset_initial_energy: float = DEFAULT_INITIAL_ENERGY,
        num_workers: int = 0, device: str = 'cpu', random_seed: int = 33
    ) -> nn.Module:

    if loss_function_kwargs is None:
        loss_function_kwargs = {}

    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    torch.manual_seed(random_seed)

    train_dataset = EnergyRecencyWeightedDataset(train_data, k=k, energy_beta=dataset_energy_beta, initial_energy=dataset_initial_energy)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    validate = val_data is not None

    min_loss = np.Inf
    patience_loss = np.Inf
    patience_update_epoch = 0
    best_model = model

    epoch = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_train_losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            X = batch[0].to(device)
            scores = model(X)
            train_dataset.update_negative_energies(scores[:, 1:])
            loss = loss_function(scores, **loss_function_kwargs)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_val_losses = []

        if validate:
            model.eval()
            with torch.no_grad():
                val_positive_scores = model(val_data.positive_samples.to(device))
                val_negative_scores = model(val_data.negative_samples.to(device))
                val_loss = val_loss_function(val_positive_scores, val_negative_scores, **loss_function_kwargs)
                epoch_val_losses.append(val_loss.item())

        if should_print and epoch % print_interval == 0:
            if validate:
                if should_print_weights:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
                else:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f}')
            else:
                if should_print_weights:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
                else:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f}')

        epoch_loss = np.mean(epoch_val_losses) if validate else np.mean(epoch_train_losses)

        if epoch_loss < min_loss:
            if should_print:
                print(f'Epoch {epoch}: new best model with loss {epoch_loss:.4f}')
            min_loss = epoch_loss
            best_model = copy.deepcopy(model).cpu()

        if epoch_loss < patience_loss - patience_threshold:
            if should_print:
                print(f'Epoch {epoch}: updating patience loss from {patience_loss:.4f} to {epoch_loss:.4f}')
            patience_loss = epoch_loss
            patience_update_epoch = epoch

        if epoch - patience_update_epoch >= patience_epochs:
            if should_print:
                print(f'Early stopping after {epoch} epochs')
            break

    if epoch == n_epochs - 1:
        print('Training finished without early stopping')

    model = best_model.to(device)

    return model


def train_and_validate_model(model: nn.Module,
    train_data: ConstrativeTrainingData,
    val_data: typing.Optional[ConstrativeTrainingData] = None,
    loss_function: typing.Callable = fitness_square_square_loss,
    loss_function_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    optimizer_class: typing.Callable = torch.optim.SGD,
    n_epochs: int = 1000, lr: float = 0.01, weight_decay: float = 0.0,
    should_print: bool = True, should_print_weights: bool = False, print_interval: int = 10,
    patience_epochs: int = 5, patience_threshold: float = 0.01,
    batch_size: int = 8, k: int = 4, device: str = 'cpu', random_seed: int = 33) -> nn.Module:

    if loss_function_kwargs is None:
        loss_function_kwargs = {}

    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    if train_data.positive_samples.shape[0] != train_data.negative_samples.shape[0]:
        raise ValueError(f'train_and_validate_model expects the number of positive samples {train_data.positive_samples.shape[0]} to match the number of negative samples {train_data.negative_samples.shape[0]}')

    train_data_tensor = torch.cat((train_data.positive_samples.unsqueeze(1), train_data.negative_samples), dim=1)

    model.to(device)
    train_dataset = TensorDataset(train_data_tensor.to(device))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate = val_data is not None
    if validate:
        if val_data.positive_samples.shape[0] != val_data.negative_samples.shape[0]:
            raise ValueError(f'train_and_validate_model expects the number of positive samples {val_data.positive_samples.shape[0]} to match the number of negative samples {val_data.negative_samples.shape[0]}')

        val_data_tensor = torch.cat((val_data.positive_samples.unsqueeze(1), val_data.negative_samples), dim=1)
        val_dataset = TensorDataset(val_data_tensor.to(device))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(random_seed)

    min_loss = np.Inf
    patience_loss = np.Inf
    patience_update_epoch = 0
    best_model = model

    epoch = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_train_losses = []
        for batch in train_dataloader:
            X = batch[0]
            optimizer.zero_grad()
            negative_indices = torch.randperm(X.shape[1] - 1)[:k] + 1
            indices = torch.cat((torch.tensor([0]), negative_indices))
            X = X[:, indices].to(device)
            scores = model(X)
            loss = loss_function(scores, **loss_function_kwargs)
            epoch_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_val_losses = []

        if validate:
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:  # type: ignore
                    X = batch[0]
                    negative_indices = torch.randperm(X.shape[1] - 1)[:k] + 1
                    indices = torch.cat((torch.tensor([0]), negative_indices))
                    X = X[:, indices].to(device)

                    scores = model(X)
                    loss = loss_function(scores, **loss_function_kwargs)
                    epoch_val_losses.append(loss.item())

        if should_print and epoch % print_interval == 0:
            if validate:
                if should_print_weights:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
                else:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f}')
            else:
                if should_print_weights:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
                else:
                    print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f}')

        epoch_loss = np.mean(epoch_val_losses) if validate else np.mean(epoch_train_losses)

        if epoch_loss < min_loss:
            if should_print:
                print(f'Epoch {epoch}: new best model with loss {epoch_loss:.4f}')
            min_loss = epoch_loss
            best_model = copy.deepcopy(model).cpu()

        if epoch_loss < patience_loss - patience_threshold:
            if should_print:
                print(f'Epoch {epoch}: updating patience loss from {patience_loss:.4f} to {epoch_loss:.4f}')
            patience_loss = epoch_loss
            patience_update_epoch = epoch

        if epoch - patience_update_epoch >= patience_epochs:
            if should_print:
                print(f'Early stopping after {epoch} epochs')
            break

    if epoch == n_epochs - 1:
        print('Training finished without early stopping')

    model = best_model.to(device)

    return model


def cross_validate(train: torch.Tensor,
    param_grid: typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]],
    scoring_function: typing.Callable = evaluate_fitness_overall_ecdf,
    scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    cv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    n_folds: int = 5, verbose: int = 0) -> GridSearchCV:

    if scaler_kwargs is None:
        scaler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if train_kwargs is None:
        train_kwargs = {}

    if cv_kwargs is None:
        cv_kwargs = {}

    if 'n_jobs' not in cv_kwargs:
        cv_kwargs['n_jobs'] = -1
    if 'verbose' not in cv_kwargs:
        cv_kwargs['verbose'] = verbose

    pipeline = Pipeline(steps=[('scaler', CustomSklearnScaler(**scaler_kwargs)), ('fitness', SklearnFitnessWrapper(model_kwargs=model_kwargs, train_kwargs=train_kwargs))])

    if isinstance(param_grid, list):
        for param_grid_dict in param_grid:
            param_grid_dict['fitness__n_features'] = [train.shape[-1]]
    else:
        param_grid['fitness__n_features'] = [train.shape[-1]]

    random_seed = train_kwargs['random_seed'] if 'random_seed' in train_kwargs else None

    cv = GridSearchCV(pipeline, param_grid, scoring=scoring_function,
        cv=KFold(n_folds, shuffle=True, random_state=random_seed),
        **cv_kwargs)
    return cv.fit(train, None)


def model_fitting_experiment(input_data: typing.Union[pd.DataFrame, torch.Tensor],
    param_grid: typing.Union[typing.List[typing.Dict[str, typing.Any]], typing.Dict[str, typing.Any]],
    split_test_set: bool = True,
    feature_columns: typing.Optional[typing.List[str]] = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    scoring_function: typing.Callable = evaluate_fitness_overall_ecdf,
    scaler_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    cv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    n_folds: int = 5, verbose: int = 0
    ) -> typing.Tuple[GridSearchCV, typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]], typing.Optional[typing.Dict[str, float]]]:

    if scaler_kwargs is None:
        scaler_kwargs = {}

    if model_kwargs is None:
        model_kwargs = {}

    if train_kwargs is None:
        train_kwargs = {}

    cv_data = input_data
    test_tensor = None

    if isinstance(cv_data, pd.DataFrame):
        if feature_columns is None:
            feature_columns = [c for c in cv_data.columns if c not in NON_FEATURE_COLUMNS]

        if split_test_set:
            cv_data, test_data = train_test_split_by_game_name(cv_data, random_seed=random_seed)
            test_tensor = df_to_tensor(test_data, feature_columns)

        cv_tensor = df_to_tensor(cv_data, feature_columns)

    else:
        cv_tensor = cv_data
        if split_test_set:
            cv_tensor, test_tensor = train_test_split(cv_tensor, random_state=random_seed,
                train_size=DEFAULT_TRAINING_PROP)

        cv_tensor = typing.cast(torch.Tensor, cv_tensor)

    if test_tensor is not None:
        print(f'Train tensor shape: {cv_tensor.shape} | Test tensor shape: {test_tensor.shape}')  # type: ignore
    else:
        print(f'Train tensor shape: {cv_tensor.shape}')

    cv = cross_validate(cv_tensor, param_grid,
        scoring_function=scoring_function,
        scaler_kwargs=scaler_kwargs, model_kwargs=model_kwargs,
        train_kwargs={'random_seed': random_seed, **train_kwargs},
        cv_kwargs=cv_kwargs, n_folds=n_folds, verbose=verbose)

    test_results = None

    if split_test_set:
        if test_tensor is None:
            raise ValueError(f'Encoutered None test tensor with split_test_set=True')
        test_tensor = typing.cast(torch.Tensor, test_tensor)
        best_model = typing.cast(SklearnFitnessWrapper, cv.best_estimator_)
        ecdf = evaluate_fitness_overall_ecdf(best_model, test_tensor)
        game_rank = evaluate_fitness_single_game_rank(best_model, test_tensor)
        test_results = dict(ecdf=ecdf, game_rank=game_rank)

    return cv, (cv_tensor, test_tensor), test_results  # type: ignore


def plot_energy_histogram(energy_model: typing.Union[GridSearchCV, Pipeline],
    train_tensor: torch.Tensor, test_tensor: typing.Optional[torch.Tensor] = None,
    histogram_title_base: str = 'Energy scores of all games',
    histogram_title_note: typing.Optional[str] = None,
    histogram_log_y: bool = True):

    if isinstance(energy_model, GridSearchCV):
        energy_model = energy_model.best_estimator_  # type: ignore

    train_positive_scores = energy_model.transform(train_tensor[:, 0, :]).detach().squeeze().numpy()  # type: ignore
    train_negative_scores = energy_model.transform(train_tensor[:, 1:, :]).detach().squeeze().numpy()  # type: ignore
    hist_scores = [train_positive_scores, train_negative_scores.flatten()]
    cm = plt.get_cmap('tab20')  # type: ignore
    colors = cm.colors[0], cm.colors[2]

    if test_tensor is not None:
        labels = ['Real (train)', 'Regrown (train)']
        test_positive_scores = energy_model.transform(test_tensor[:, 0, :]).detach().squeeze().numpy()  # type: ignore
        test_negative_scores = energy_model.transform(test_tensor[:, 1:, :]).detach().squeeze().numpy()  # type: ignore

        hist_scores.insert(1, test_positive_scores)
        hist_scores.append(test_negative_scores.flatten())
        labels.insert(1, 'Real (test)')
        labels.append('Regrown (test)')

        colors = cm.colors[:4]

    else:
        labels = ['Real', 'Regrown']

    plt.hist(hist_scores, label=labels, stacked=True, bins=100, color=colors)  # type: ignore
    if histogram_title_note is not None:
        plt.title(f'{histogram_title_base} ({histogram_title_note})')
    else:
        plt.title(histogram_title_base)

    plt.xlabel('Energy score')

    if histogram_log_y:
        plt.ylabel('log(Count)')
        plt.semilogy()
    else:
        plt.ylabel('Count')

    plt.legend(loc='best')
    plt.show()


def visualize_cv_outputs(cv: GridSearchCV, train_tensor: torch.Tensor,
    test_tensor: typing.Optional[torch.Tensor] = None,
    test_results: typing.Optional[dict] = None,
    display_by_ecdf: bool = True, display_by_game_rank: bool = True,
    display_energy_histogram: bool = True, histogram_title_base: str = 'Energy scores of all games',
    histogram_title_note: typing.Optional[str] = None, histogram_log_y: bool = True,
    dispaly_weights_histogram: bool = True, weights_histogram_title_base: str = 'Energy model weights',
    ) -> None:

    cv_df = pd.concat([
        pd.DataFrame(cv.cv_results_["params"]),
        pd.DataFrame(cv.cv_results_["mean_test_overall_ecdf"], columns=['ecdf_mean']),
        pd.DataFrame(cv.cv_results_["std_test_overall_ecdf"], columns=['ecdf_std']),
        pd.DataFrame(cv.cv_results_["rank_test_overall_ecdf"], columns=['ecdf_rank']),
        pd.DataFrame(cv.cv_results_["mean_test_single_game_rank"], columns=['game_rank_mean']),
        pd.DataFrame(cv.cv_results_["std_test_single_game_rank"], columns=['game_rank_std']),
        pd.DataFrame(cv.cv_results_["rank_test_single_game_rank"], columns=['game_rank_rank']),
    ],axis=1)

    if test_results is not None:
        display(Markdown('### Test results:'))
        display(test_results)

    if display_by_ecdf:
        display(Markdown('### CV results by overall ECDF:'))
        display(cv_df.sort_values(by='ecdf_rank').head(10))

    if display_by_game_rank:
        display(Markdown('### CV results by mean single game rank:'))
        display(cv_df.sort_values(by='game_rank_rank').head(10))

    if display_energy_histogram:
        plot_energy_histogram(cv, train_tensor, test_tensor, histogram_title_base, histogram_title_note, histogram_log_y)

    if dispaly_weights_histogram:
        weights = cv.best_estimator_.named_steps['fitness'].model.fc1.weight.data.detach().numpy().squeeze()  # type: ignore
        bias = cv.best_estimator_.named_steps['fitness'].model.fc1.bias.data.detach().numpy().squeeze()  # type: ignore
        print(weights.mean(), weights.std(), bias)

        plt.hist(weights, bins=100)

        if histogram_title_note is not None:
            plt.title(f'{weights_histogram_title_base} ({histogram_title_note})')
        else:
            plt.title(weights_histogram_title_base)

        plt.xlabel('Weight magnitude')
        plt.ylabel('Count')
        plt.show()



HTML_DIFF = HtmlDiff(wrapcolumn=100)
HTML_DIFF_SUBSTITUTIONS = {
    'td.diff_header {text-align:right}': '.diff td {text-align: left !important}\n.diff th {text-align: center!important }\n.diff td.diff_header {text-align:right !important}',
    '.diff_add {background-color:#aaffaa}': '.diff_add {background-color: #6fa66f !important; font-weight: bold !important}',
    '.diff_chg {background-color:#ffff77}': '.diff_chg {background-color: #999949 !important; font-weight: bold !important}',
    '.diff_sub {background-color:#ffaaaa}': '.diff_sub {background-color: #a66f6f !important; font-weight: bold !important}',
    # '.diff_add {background-color:#aaffaa}': '.diff_add {color: #6fa66f; background-color: inherit; font-weight: bold}',
    # '.diff_chg {background-color:#ffff77}': '.diff_chg {color: #999949; background-color: inherit; font-weight: bold}',
    # '.diff_sub {background-color:#ffaaaa}': '.diff_sub {color: #a66f6f; background-color: inherit; font-weight: bold}',
}


def display_game_diff_html(before: str, after: str, html_diff_substitutions: typing.Dict[str, str] = HTML_DIFF_SUBSTITUTIONS):
    diff = HTML_DIFF.make_file(before.splitlines(), after.splitlines())  #, context=True, numlines=0)

    for key, value in html_diff_substitutions.items():
        diff = diff.replace(key, value)

    display(HTML(diff))


def evaluate_single_game_energy_contributions(cv: typing.Union[GridSearchCV, Pipeline], game_features: torch.Tensor, game_text: str,
    feature_names: typing.List[str], top_k: int = 20, display_overall_features: bool = True, display_game: bool = True, min_display_threshold: float = 0.0005,):

    energy_model = cv
    if isinstance(cv, GridSearchCV):
        energy_model = cv.best_estimator_
    weights = energy_model['fitness'].model.fc1.weight.data.detach().squeeze()  # type: ignore

    index_energy_contributions = game_features * weights

    real_game_energy = energy_model.transform(game_features).item()  # type: ignore

    display(Markdown(f'### Energy of visualized game: {real_game_energy:.3f}'))

    if display_overall_features:
        display_energy_contributions_table(index_energy_contributions, game_features, weights, feature_names, top_k, min_display_threshold)

    if display_game:
        display(Markdown(f'### Game:'))
        display(Markdown(f'```pddl\n{game_text}\n```'))


def evaluate_energy_contributions(energy_model: typing.Union[GridSearchCV, Pipeline], data_tensor: torch.Tensor, index: typing.Union[int, typing.Tuple[int, int]],
    feature_names: typing.List[str], full_dataset_tensor: torch.Tensor,
    original_game_texts: typing.List[str], negative_game_texts: typing.List[str],
    index_in_negatives: bool = True, top_k: int = 10, display_overall_features: bool = False, display_relative_features: bool = True,
    display_features_pre_post_scaling: bool = False,
    display_game_diff: bool = True, html_diff_substitutions: typing.Dict[str, str] = HTML_DIFF_SUBSTITUTIONS, min_display_threshold: float = 0.0005,
    display_features_diff: bool = True) -> None:

    negatives = data_tensor[:, 1:, :]
    if isinstance(index, tuple):
        row, col = index
    else:
        if index_in_negatives:
            row, col = torch.div(index, negatives.shape[1], rounding_mode='trunc'), index % negatives.shape[1]
        else:
            row, col = torch.div(index, full_dataset_tensor.shape[1], rounding_mode='trunc') , index % full_dataset_tensor.shape[1]

    if index_in_negatives:
        index_features = negatives[row, col]
    else:
        index_features = full_dataset_tensor[row, col]

    real_game_features = data_tensor[row, 0]

    if index_in_negatives:
        original_game_index = (full_dataset_tensor[:, :2, :] == data_tensor[row, :2, :]).all(dim=-1).all(dim=-1).nonzero().item()
        print(f'Original game index: {original_game_index} | Negative game row: {row} | Negative game col: {col}')
    else:
        original_game_index = index

    original_game_text = original_game_texts[original_game_index]  # type: ignore
    negative_game_text = negative_game_texts[(original_game_index * negatives.shape[1]) + col]  # type: ignore

    evaluate_comparison_energy_contributions(
        real_game_features, index_features, original_game_text,
        negative_game_text, energy_model, feature_names, top_k,
        display_overall_features, display_relative_features, display_game_diff,
        html_diff_substitutions, min_display_threshold, display_features_diff
    )

def evaluate_comparison_energy_contributions(
    original_game_features: torch.Tensor, comparison_game_features: torch.Tensor,
    original_game_text: str, comparison_game_text: str,
    energy_model: typing.Union[GridSearchCV, Pipeline],
    feature_names: typing.List[str], top_k: int = 10, display_overall_features: bool = False, display_relative_features: bool = True,
    display_game_diff: bool = True, html_diff_substitutions: typing.Dict[str, str] = HTML_DIFF_SUBSTITUTIONS, min_display_threshold: float = 0.0005,
    display_features_diff: bool = True
    ):

    if isinstance(energy_model, GridSearchCV):
        energy_model = energy_model.best_estimator_  # type: ignore

    index_energy = energy_model.transform(comparison_game_features).item()  # type: ignore
    real_game_energy = energy_model.transform(original_game_features).item()  # type: ignore

    weights = energy_model['fitness'].model.fc1.weight.data.detach().squeeze()  # type: ignore

    scaled_index_features = energy_model['scaler'].transform(comparison_game_features)  # type: ignore
    scaled_real_game_features = energy_model['scaler'].transform(original_game_features)  # type: ignore

    index_energy_contributions = scaled_index_features * weights
    real_game_contributions = scaled_real_game_features * weights

    display(Markdown(f'### Energy of real game: {real_game_energy:.3f} | Energy of regrown game: {index_energy:.3f} | Difference: {index_energy - real_game_energy:.3f}'))

    if display_overall_features:
        display_energy_contributions_table(index_energy_contributions, comparison_game_features, weights, feature_names, top_k, min_display_threshold)

    if display_relative_features:
        relative_contributions = index_energy_contributions - real_game_contributions
        display_energy_contributions_table(relative_contributions, comparison_game_features, weights, feature_names, top_k, min_display_threshold, original_game_features)

    if display_game_diff:
        display(Markdown('### Game Diffs'))
        display_game_diff_html(original_game_text, comparison_game_text, html_diff_substitutions)

    if display_features_diff:
        display(Markdown('### Feature Diffs'))
        d = comparison_game_features - original_game_features
        inds = d.nonzero().squeeze()
        if inds.ndim == 0 or len(inds) == 0:
            print('No features changed')

        else:
            diffs = d[inds]
            for i in torch.argsort(diffs):
                original_idx = inds[i]
                print(f'{feature_names[original_idx]}: {diffs[i]:.3f} ({scaled_real_game_features[original_idx]:.3f} => {scaled_index_features[original_idx]:.3f})')


def display_energy_contributions_table(energy_contributions: torch.Tensor, feature_values: torch.Tensor, weights: torch.Tensor,
        feature_names: typing.List[str], top_k: int, min_display_threshold: float = 0.005, real_game_features: typing.Optional[torch.Tensor] = None):
    energy_up_features = []
    energy_down_features = []

    top_k_contributions = torch.topk(energy_contributions, top_k, largest=True)
    if torch.any(top_k_contributions.values > min_display_threshold):
        for i in range(top_k):
            idx = top_k_contributions.indices[i]
            value = top_k_contributions.values[i]
            if value > min_display_threshold:
                # if display_features_pre_post_scaling:
                #     energy_up_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {scaled_real_game_features[idx]:.3f} | {index_features[idx]:.3f} => {scaled_index_features[idx]:.3f}) * {weights[idx]:.3f}')
                # else:
                if real_game_features is not None:
                    energy_up_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {feature_values[idx]:.3f}) * {weights[idx]:.3f}')
                else:
                    energy_up_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({feature_values[idx]:.3f}) * {weights[idx]:.3f}')

    bottom_k_contributions = torch.topk(energy_contributions, top_k, largest=False)
    if torch.any(bottom_k_contributions.values < -min_display_threshold):
        for i in range(top_k):
            idx = bottom_k_contributions.indices[i]
            value = bottom_k_contributions.values[i]
            if value < -min_display_threshold:
                # if display_features_pre_post_scaling:
                #     energy_down_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {scaled_real_game_features[idx]:.3f} | {index_features[idx]:.3f} => {scaled_index_features[idx]:.3f}) * {weights[idx]:.3f}')
                # else:
                if real_game_features is not None:
                    energy_down_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({real_game_features[idx]:.3f} => {feature_values[idx]:.3f}) * {weights[idx]:.3f}')
                else:
                    energy_down_features.append(f'{feature_names[idx]}: **{value:.3f}** = ({feature_values[idx]:.3f}) * {weights[idx]:.3f}')

    if real_game_features is not None:
        display(Markdown(f'### Top features changing the game\'s energy\nfeature name: **value** = (original feature value => regrown feature value) * weight'))
    else:
        display(Markdown(f'### Top features contributing to the game\'s energy\nfeature name: **value** = (original feature value => regrown feature value) * weight'))

    rows = list(zip_longest(energy_up_features, energy_down_features))
    headers = ['Features increasing energy (= more fake)', 'Features decreasing energy (= more real)']
    table = tabulate(rows, headers=headers, tablefmt='github')
    display(Markdown(table))
