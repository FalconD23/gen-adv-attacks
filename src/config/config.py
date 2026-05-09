import os
import shutil
from typing import Dict, Iterable, List

import torch
from src import attacks, estimation, models
from src.attacks import attack_scheduler
from src.utils import weights_from_clearml_by_name
from src import iter_strategies

from src.optim.muon import SingleDeviceMuonWithAuxAdam


def get_attack(attack_name: str, attack_params: Dict) -> attacks.BaseIterativeAttack:
    if attack_params is None:
        attack_params = dict()
    try:
        return getattr(attacks, attack_name)(**attack_params)
    except AttributeError:
        raise ValueError(f"Attack with name {attack_name} is not implemented")



def get_attacker_model(
    surrogate_model_name: str,
    attacker_model_params: Dict,
    device: str = "cpu",
    path: str = None,
    train_mode: bool = False,
) -> torch.nn.Module:
    """
    Get a surrogate model for MBA attacks.
    
    Args:
        surrogate_model_name: Name of the surrogate model class
        attacker_model_params: Parameters for the surrogate model
        device: Device to load the model on
        path: Path to load the model weights from
        train_mode: Whether to set the model to training mode
        
    Returns:
        The surrogate model
    """
    if attacker_model_params is None:
        attacker_model_params = dict()
    try:
        model = getattr(models, surrogate_model_name)(**attacker_model_params)
        model = model.to(device)
        if path:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        model.train(train_mode)
        return model
    except AttributeError:
        raise ValueError(f"Surrogate model with name {surrogate_model_name} is not implemented")


def get_model(
    model_name: str,
    model_params: Dict,
    device: str = "cpu",
    path: str = None,
    train_mode: bool = False,
) -> torch.nn.Module:
    if model_params is None:
        model_params = dict()
    try:
        model = getattr(models, model_name)(**model_params)
        model = model.to(device)
        if path:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=torch.device(device)))
                print(f"Successfully loaded model weights from {path}")
            else:
                print(f"Warning: Model weights file not found at {path}. Creating model from scratch.")
        model.train(train_mode)
        return model
    except AttributeError:
        raise ValueError(f"Model with name {model_name} is not implemented")


def get_criterion(
    criterion_name: str, criterion_params: Dict = None
) -> torch.nn.Module:
    if criterion_params is None:
        criterion_params = dict()
    try:
        return getattr(torch.nn, criterion_name)(**criterion_params)
    except AttributeError:
        raise ValueError(f"Criterion with name {criterion_name} is not implemented")


def get_optimizer(
    optimizer_name: str, model_params: Dict, optimizer_params: Iterable = None
):
    if optimizer_params is None:
        optimizer_params = dict()

    params = [p for p in model_params if p.requires_grad]
    if optimizer_name == "Muon":
        # Safety-first: hybrid mode (Muon for matrix params, Adam-like aux for scalars/bias/norm)
        cfg = dict(optimizer_params)
        muon_lr = cfg.pop("lr", 0.003)
        muon_momentum = cfg.pop("momentum", 0.95)
        muon_weight_decay = cfg.pop("weight_decay", 0.01)
        aux_lr = cfg.pop("aux_lr", 1e-2)
        aux_betas = cfg.pop("aux_betas", (0.9, 0.95))
        aux_eps = cfg.pop("aux_eps", 1e-8)
        aux_weight_decay = cfg.pop("aux_weight_decay", 0.01)
        if cfg:
            raise ValueError(f"Unknown Muon optimizer params: {list(cfg.keys())}")
        muon_params = [p for p in params if p.ndim >= 2]
        aux_params = [p for p in params if p.ndim < 2]
        param_groups = []
        if muon_params:
            param_groups.append(
                dict(
                    params=muon_params,
                    lr=muon_lr,
                    momentum=muon_momentum,
                    weight_decay=muon_weight_decay,
                    use_muon=True,
                )
            )
        if aux_params:
            param_groups.append(
                dict(
                    params=aux_params,
                    lr=aux_lr,
                    betas=aux_betas,
                    eps=aux_eps,
                    weight_decay=aux_weight_decay,
                    use_muon=False,
                )
            )
        if not param_groups:
            raise ValueError("No trainable parameters found for Muon optimizer.")
        return SingleDeviceMuonWithAuxAdam(param_groups)


    try:
        return getattr(torch.optim, optimizer_name)(params, **optimizer_params)
    except AttributeError:
        raise ValueError(f"Optimizer with name {optimizer_name} is not implemented")


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, scheduler_params: Dict = None
) -> torch.optim.lr_scheduler.LRScheduler:
    if scheduler_name is None or scheduler_name == "None":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    if scheduler_params is None:
        scheduler_params = dict()
    try:
        return getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer, **scheduler_params
        )
    except AttributeError:
        raise ValueError(f"Scheduler with name {scheduler_name} is not implemented")


def get_attack_scheduler(
    attack_scheduler_name: str,
    attack: attacks.BaseIterativeAttack,
    attack_scheduler_params: Dict = None,
) -> attack_scheduler.AttackScheduler:
    if attack_scheduler_params is None:
        attack_scheduler_params = dict()
    try:
        return getattr(attack_scheduler, attack_scheduler_name)(
            attack, **attack_scheduler_params
        )
    except AttributeError:
        raise ValueError(
            f"Attack Scheduler with name {attack_scheduler_name} is not implemented"
        )


def get_disc_list(
    model_name: str,
    model_params: Dict,
    list_disc_params: List[Dict],
    device: str = "cpu",
    path: str = "",
    train_mode: bool = False,
    from_clearml: bool = False
):
    list_disc_models = list()
    for model_conf in list_disc_params:
        if isinstance(model_conf, dict):
            weight_name = model_conf['model_id']
            model_folder_name = model_conf['model_name']
        else:
            weight_name = model_conf
            model_folder_name = model_conf
        disc_path = f"{path}/{model_folder_name}/{weight_name}.pt"
        folder_path = 'loaded_clearml/disc_weights'
        file_path = os.path.join(folder_path, weight_name)
        if from_clearml and os.path.exists(file_path):
            loaded_path = file_path
        elif from_clearml:
            loaded_path = weights_from_clearml_by_name(path, f"{weight_name}", load_weights='disc_weights')


        if from_clearml:
            disc = get_model(
            model_name,
            model_params,
            device=device,
            path=loaded_path,
            train_mode=train_mode,
        )
        else:
            disc = get_model(
                model_name,
                model_params,
                device=device,
                path=disc_path,
                train_mode=train_mode,
            )
        list_disc_models.append(disc)
    return list_disc_models


def get_iter_strategy(name: str,
                      attack: attacks.BaseIterativeAttack,
                      params: dict = None):
    if params is None:
        params = {}

    if name in iter_strategies.PRESETS:
        return iter_strategies.PRESETS[name](attack, **params)
    try:
        return getattr(iter_strategies, name)(attack, **params)
    except KeyError:
        raise ValueError(f"IterStrategy {name} not implemented")
