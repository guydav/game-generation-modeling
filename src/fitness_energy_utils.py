from collections import defaultdict
import copy
import typing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


FITNESS_DATA_FILE = '../data/fitness_scores.csv'
NON_FEATURE_COLUMNS = set(['Index', 'src_file', 'game_name', 'domain_name', 'real', 'original_game_name'])


def load_fitness_data(path: str = FITNESS_DATA_FILE) -> pd.DataFrame:
    fitness_df = pd.read_csv(FITNESS_DATA_FILE)
    fitness_df = fitness_df.assign(real=fitness_df.src_file == 'interactive-beta.pddl', original_game_name=fitness_df.game_name)
    fitness_df.original_game_name.where(
        fitness_df.game_name.apply(lambda s: s.count('-') <= 1), 
        fitness_df.original_game_name.apply(lambda s: s[:s.rfind('-')]), 
        inplace=True)

    return fitness_df


DEFAULT_RANDOM_SEED = 33
DEFAULT_TRAINING_PROP = 0.8


def train_test_split_by_game_name(df: pd.DataFrame, training_prop: float = DEFAULT_TRAINING_PROP,
    random_seed: int = DEFAULT_RANDOM_SEED, positive_column: str = 'real', positive_value: typing.Any = True):

    real_game_names = df[df[positive_column] == positive_value].game_name.unique()

    train_game_names, test_game_names = train_test_split(real_game_names, train_size=training_prop, random_state=random_seed)
    train_df = df[df.game_name.isin(train_game_names) | df.original_game_name.isin(train_game_names)]
    test_df = df[df.game_name.isin(test_game_names) | df.original_game_name.isin(test_game_names)]
    return train_df, test_df


def df_to_tensor(df: pd.DataFrame, feature_columns: typing.List[str], 
    positive_column: str = 'real', positive_value: typing.Any = True):
    return torch.tensor(
        np.stack([
            np.concatenate((
                df.loc[df.game_name == game_name, feature_columns].to_numpy(),
                df.loc[(df.game_name != game_name) & (df.original_game_name == game_name), feature_columns].to_numpy()
            ))
            for game_name
            in df[df[positive_column] == positive_value].game_name.unique()
        ]),
        dtype=torch.float
    )


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)



class CustomSklearnScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        if X.ndim != 3:
            raise ValueError('X must be 3D')

        self.mean = X.mean(axis=(0, 1))
        self.std = X.std(axis=(0, 1))   
        return self

    def transform(self, X, y=None):
        if X.ndim != 3:
            raise ValueError('X must be 3D')

        return (X - self.mean) / self.std
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
        
    def get_feature_names_out(self, input_features=None):
        return [f'x{i}' for i in range(self.mean.shape[0])]  # type: ignore

    def set_params(self, **params):
        return self

    def get_params(self, deep=True):
        return {}


class FitnessEenrgyModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: typing.Optional[int] = None,
        hidden_activation: typing.Callable = torch.relu,
        n_outputs: int = 1):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs

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

        # TODO: do we want a sigmoid or something else? Or nothing at all? 
        if self.n_outputs == 1 and activate:
            x = torch.sigmoid(x)

        return x


DEFAULT_MODEL_PARAMS = {
    'n_features': None,
    'hidden_size': None,
    'hidden_activation': torch.relu,
    'n_outputs': 1,
}

DEFAULT_TRAIN_KWARGS = {
    'weight_decay': 0.0,
    'lr': 1e-2,
    'should_print': False, 
    'print_interval': 10,
    'patience_epochs': 5, 
    'patience_threshold': 0.01, 
    'batch_size': 8, 
    'k': 4, 
    'device': 'cpu',
    'seed': 33,
}

class SklearnFitnessWrapper:
    def __init__(self,
        model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, 
        train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, **params):

        self.model_kwargs = copy.deepcopy(DEFAULT_MODEL_PARAMS)
        if model_kwargs is not None:
            self.model_kwargs.update(model_kwargs)

        self.train_kwargs = copy.deepcopy(DEFAULT_TRAIN_KWARGS)
        if train_kwargs is not None:
            self.train_kwargs.update(train_kwargs)

        self.set_params(**params)

    def get_params(self, deep: bool = True):
        return {
            **self.model_kwargs,
            **self.train_kwargs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.model_kwargs:
                self.model_kwargs[key] = value
            elif key in self.train_kwargs:
                self.train_kwargs[key] = value
            else:
                raise ValueError(f'Unknown parameter {key}')

        return self

    def fit(self, X, y=None):
        self.model = FitnessEenrgyModel(**self.model_kwargs)
        self.model.apply(init_weights)
        self.model = train_and_validate_model(self.model, X, **self.train_kwargs)[0] # type: ignore
        return self
            
    def __call__(self, *args, **kwargs):
        if self.model is not None:
            return self.model(*args, **kwargs)

        return None
        

def nce_fitness_loss(scores: torch.Tensor):
    if scores.shape[-1] == 1:
        positive_scores = torch.log(scores[:, 0])
        negative_scores = torch.log(1 - scores[:, 1:]).sum(axis=1)  # type: ignore
    else:
        positive_scores = torch.log(scores[:, 0, 0])
        negative_scores = torch.log(1 - scores[:, 1:, 1]).sum(axis=1)  # type: ignore
        
    return -(positive_scores + negative_scores).mean()


def evaluate_fitness(model: typing.Union[nn.Module, SklearnFitnessWrapper], X: torch.Tensor, y=None, return_all=False):
    return_all = return_all and isinstance(model, nn.Module)
    
    if isinstance(model, Pipeline):
        model = model.named_steps['fitness']
    
    if isinstance(model, SklearnFitnessWrapper):
        model = model.model

    model.eval()
    with torch.no_grad():
        scores = model(X, activate=False)
        if scores.shape[-1] == 1:
            positive_scores = scores[:, 0]
            negative_scores = scores[:, 1:]
        else:
            positive_scores = scores[:, 0, 0]
            negative_scores = scores[:, 1:, 1]
        game_average_scores = positive_scores - negative_scores.mean(axis=1)
        if return_all:
            return positive_scores.mean(), negative_scores.mean(), game_average_scores.mean()
        else:
            return game_average_scores.mean().item()


def train_and_validate_model(model: nn.Module, 
    train_data: torch.Tensor, 
    val_data: typing.Optional[torch.Tensor] = None, 
    n_epochs: int = 100, lr: float = 0.01, weight_decay: float = 0.0, 
    should_print: bool = True, print_interval: int = 10,
    patience_epochs: int = 5, patience_threshold: float = 0.01, 
    batch_size: int = 8, k: int = 4, device: str = 'cpu', seed: int = 33):

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate = val_data is not None
    if validate:
        val_dataset = TensorDataset(val_data)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(seed)

    min_loss = np.Inf
    patience_loss = np.Inf
    patience_update_epoch = 0
    best_model = model
    
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
            loss = nce_fitness_loss(scores)
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
                    loss = nce_fitness_loss(scores)
                    epoch_val_losses.append(loss.item())

        if should_print and epoch % print_interval == 0:
            if validate:
                print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | val loss {np.mean(epoch_val_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore
            else:
                print(f'Epoch {epoch}: train loss {np.mean(epoch_train_losses):.4f} | weights {model.fc1.weight.data}')  # type: ignore

        epoch_loss = np.mean(epoch_val_losses) if validate else np.mean(epoch_train_losses)

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model = copy.deepcopy(model).cpu()

        if epoch_loss < patience_loss - patience_threshold:
            patience_loss = epoch_loss
            patience_update_epoch = epoch

        if epoch - patience_update_epoch >= patience_epochs:
            break

    model = best_model.to(device)

    if validate:    
        return model, evaluate_fitness(model, train_data), evaluate_fitness(model, val_data)
    else:
        return model, evaluate_fitness(model, train_data)


