import jax.numpy as jnp
from jax import lax
import jax
from enum import IntEnum
from jaxatari.games.jax_amidar import AmidarState, JaxAmidar

GAMMA = 0.99 

def unpack(state):
    while not isinstance(state, AmidarState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a AmidarState or does not contain a AmidarState.")
    return state

def env_reward(prev_state: AmidarState, state: AmidarState) -> float:
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Compute the environment reward based on the previous and current state
    return JaxAmidar()._get_env_reward(prev_state, state)