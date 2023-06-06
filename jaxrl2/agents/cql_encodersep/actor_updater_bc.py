from statistics import mean
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from jaxrl2.types import Params, PRNGKey


def log_prob_update_bc(rng: PRNGKey, actor: TrainState, batch: FrozenDict):

    rng, key, key_act = jax.random.split(rng, 3)

    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Any, Dict[str, float]]]:
        
        if actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats},
                              batch['observations'], training=True, rngs={'dropout': key}, mutable=['batch_stats'])
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'], training=True, rngs={'dropout': key})
            new_model_state = {}
        
        # clip actions to be in [-1, 1]
        actions = batch['actions']
        eps = 1e-6
        actions = jnp.clip(actions, -1.0 + eps, 1.0 - eps)
            
        log_probs = dist.log_prob(actions)
        log_prob_loss = - (log_probs).mean()

        mse = (dist.mode() - actions) ** 2
        mse = mse.mean(axis=-1) # mean over action dimension
        mse_loss = (mse).mean()
                
        actor_loss = log_prob_loss

        # sample log pis for entropy calculation
        _, log_pi = dist.sample_and_log_prob(seed=key_act)
    
        if hasattr(dist, 'distribution') and hasattr(dist.distribution, '_loc'):
            mean_dist = dist.distribution._loc
            std_diag_dist = dist.distribution._scale_diag
        else:
            mean_dist = dist._loc
            std_diag_dist = dist._scale_diag
            
        things_to_log={        
            'log_prob_loss': log_prob_loss,
            'mse_loss': mse_loss, 
            'log_pi': log_pi.mean(),

            'dataset_actions': batch['actions'],
            'pred_actions_mean': mean_dist, 
            'action_std': std_diag_dist,
            'loss': actor_loss,

            'entropy': -log_pi.mean(),
        }

        return actor_loss, (new_model_state, things_to_log)

    grads, (new_model_state, info) = jax.grad(loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)
        
    return new_actor, info