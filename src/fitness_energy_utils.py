from collections import defaultdict
import copy
import typing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from statsmodels.distributions.empirical_distribution import ECDF
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


FITNESS_DATA_FILE = '../data/fitness_scores.csv'
NON_FEATURE_COLUMNS = set(['Index', 'src_file', 'game_name', 'domain_name', 'real', 'original_game_name'])


def load_fitness_data(path: str = FITNESS_DATA_FILE) -> pd.DataFrame:
    fitness_df = pd.read_csv(path)
    fitness_df = fitness_df.assign(real=fitness_df.src_file == 'interactive-beta.pddl', original_game_name=fitness_df.game_name)
    fitness_df.original_game_name.where(
        fitness_df.game_name.apply(lambda s: (s.count('-') <= 1) or (s.startswith('game-id') and s.count('-') == 2)), 
        fitness_df.original_game_name.apply(lambda s: s[:s.rfind('-')]), 
        inplace=True)

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
        self.std[torch.isclose(self.std, torch.zeros_like(self.std))] = 1
        return self

    def transform(self, X, y=None):
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


def fitness_hinge_loss(scores: torch.Tensor, margin: float = 1.0, negative_score_reduction: str = 'mean', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    if negative_score_reduction == 'none':
        positive_scores = positive_scores.unsqueeze(-1)
    return _reduce(torch.relu(positive_scores + margin - negative_scores), reduction)
    

def fitness_hinge_loss_with_cross_example(scores: torch.Tensor, margin: float = 1.0, alpha: float = 0.5,
    negative_score_reduction: str = 'mean', reduction: str = 'mean'):
    hinge = fitness_hinge_loss(scores, margin, negative_score_reduction, reduction)

    positive_scores = scores[:, 0, None]
    negative_scores = scores[:, 1:]
    cross_example_loss = _reduce(torch.relu(positive_scores + margin - negative_scores), reduction)

    return alpha * hinge + (1 - alpha) * cross_example_loss


def fitness_log_loss(scores: torch.Tensor, negative_score_reduction: str = 'mean', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    # negative_scores = scores[:, 1:].sum(dim=1)  
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    return _reduce(torch.log(1 + torch.exp(positive_scores - negative_scores)), reduction)


def fitness_square_square_loss(scores: torch.Tensor, margin: float = 1.0, negative_score_reduction: str = 'mean', reduction: str = 'mean'):
    positive_scores = scores[:, 0]
    # negative_scores = scores[:, 1:].sum(dim=1)  
    negative_scores = _reduce(scores[:, 1:], negative_score_reduction, dim=1)
    return _reduce(positive_scores.pow(2) + torch.relu(margin - negative_scores).pow(2), reduction)


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
        self.model = FitnessEenrgyModel(**self.model_kwargs)
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



def _evaluate_fitness(model: ModelClasses, X: torch.Tensor, y: typing.Optional[torch.Tensor], 
    device: str = 'cpu') -> typing.Tuple[torch.Tensor, torch.Tensor]:

    with torch.no_grad():
        if isinstance(model, Pipeline):
            model.named_steps['fitness'].model.eval()
            scores = model.transform(X.to(device))
        
        elif isinstance(model, SklearnFitnessWrapper):
            model.model.eval()
            scores = model.transform(X.to(device))

        else:
            model.eval()
            scores = model(X.to(device), activate=False)

        scores = scores.detach().cpu()

    positive_scores = scores[:, 0]
    negative_scores = scores[:, 1:]
    return positive_scores.detach(), negative_scores.detach()


# def evaluate_fitness(model: ModelClasses, X: torch.Tensor, y: typing.Optional[torch.Tensor] = None, 
#     score_sign: int = 1):
#     positive_scores, negative_scores = _evaluate_fitness(model, X, y)
        
#     game_average_scores = (positive_scores - negative_scores.mean(dim=1)) * score_sign
#     return game_average_scores.mean().item()


# def evaluate_fitness_flipped_sign(model: ModelClasses, 
#     X: torch.Tensor, y=None):
#     return evaluate_fitness(model, X, y, score_sign=-1)


def evaluate_fitness_overall_ecdf(model: ModelClasses, 
    X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _evaluate_fitness(model, X, y)
    positive_scores = positive_scores.squeeze().cpu().numpy()
    negative_scores = negative_scores.squeeze().cpu().numpy()
    ecdf = ECDF(np.concatenate([positive_scores, negative_scores.reshape(-1)]))

    positive_mean_quantile = ecdf(positive_scores).mean()
    return -positive_mean_quantile


def evaluate_fitness_single_game_rank(model: ModelClasses, X: torch.Tensor, y=None) -> float:
    positive_scores, negative_scores = _evaluate_fitness(model, X, y)
    single_game_rank = (positive_scores[:, None] < negative_scores).float().mean(axis=1)  # type: ignore
    return single_game_rank.mean().item()


def build_multiple_scoring_function(
    evaluators: typing.Sequence[typing.Callable[[ModelClasses, torch.Tensor, typing.Optional[torch.Tensor]], float]],
    names: typing.Sequence[str]
    ) -> typing.Callable[[ModelClasses, torch.Tensor, typing.Optional[torch.Tensor]], typing.Dict[str, float]]:
    def _evaluate_fitness_multiple(model: ModelClasses, X: torch.Tensor, y=None, return_all=False):
        return {name: evaluator(model, X, y) for name, evaluator in zip(names, evaluators)}

    return _evaluate_fitness_multiple


def train_and_validate_model(model: nn.Module, 
    train_data: torch.Tensor, 
    val_data: typing.Optional[torch.Tensor] = None, 
    loss_function: typing.Callable = fitness_nce_loss,
    loss_function_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    optimizer_class: typing.Callable = torch.optim.SGD,
    n_epochs: int = 1000, lr: float = 0.01, weight_decay: float = 0.0, 
    should_print: bool = True, should_print_weights: bool = False, print_interval: int = 10,
    patience_epochs: int = 5, patience_threshold: float = 0.01, 
    batch_size: int = 8, k: int = 4, device: str = 'cpu', random_seed: int = 33) -> nn.Module:

    if loss_function_kwargs is None:
        loss_function_kwargs = {}

    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validate = val_data is not None
    if validate:
        val_dataset = TensorDataset(val_data)
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
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None, 
    cv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    n_folds: int = 5, verbose: int = 0) -> GridSearchCV:

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

    pipeline = Pipeline(steps=[('scaler', CustomSklearnScaler()), ('fitness', SklearnFitnessWrapper(model_kwargs=model_kwargs, train_kwargs=train_kwargs))])

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
    model_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    train_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    cv_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    n_folds: int = 5, verbose: int = 0
    ) -> typing.Tuple[GridSearchCV, typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]], typing.Optional[typing.Dict[str, float]]]:

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

    cv = cross_validate(cv_tensor, param_grid,   
        scoring_function=scoring_function,
        train_kwargs={'random_seed': random_seed, **train_kwargs}, 
        model_kwargs=model_kwargs, cv_kwargs=cv_kwargs, n_folds=n_folds, verbose=verbose)
    
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
