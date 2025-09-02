from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.lax as lax
from serl_launcher.common.common import default_init
import jax

class STN(nn.Module):
    dim: int
    stn_bottle_neck: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        B = x.shape[0]
        P = x.shape[1]
        
        x = nn.Conv(features=64, kernel_size=1)(x) # (B, P, dim) -> (B, P, 64)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=1)(x) # (B, P, 64) -> (B, P, 128)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=self.stn_bottle_neck, kernel_size=1)(x) # (B, P, 128) -> (B, P, self.stn_bottle_neck)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # x = nn.max_pool(x, window_shape=(P,)).reshape(B, self.stn_bottle_neck) # (B, P, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        x = jnp.max(x, axis=1).reshape(B, self.stn_bottle_neck) # (B, P, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        
        x = nn.Dense(features=self.stn_bottle_neck)(x) # (B, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=256)(x) # (B, self.stn_bottle_neck) -> (B, 256)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.dim**2)(x).reshape(B, self.dim, self.dim) # (B, 512) -> (B, dim^2) -> (B, dim, dim)
        
        eye = jnp.tile(jnp.eye(self.dim).reshape(1, self.dim, self.dim), (B, 1, 1))
        eye = jax.device_put(eye)
        
        x = x + eye
    
        return x
    
class STN_LN(nn.Module):
    dim: int
    stn_bottle_neck: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        B = x.shape[0]
        P = x.shape[1]
        
        x = nn.Conv(features=64, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, dim) -> (B, P, 64)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 64) -> (B, P, 128)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=self.stn_bottle_neck, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 128) -> (B, P, self.stn_bottle_neck)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        # x = nn.max_pool(x, window_shape=(P,)).reshape(B, self.stn_bottle_neck) # (B, P, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        x = jnp.max(x, axis=1).reshape(B, self.stn_bottle_neck) # (B, P, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        
        x = nn.Dense(features=self.stn_bottle_neck)(x) # (B, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=256)(x) # (B, self.stn_bottle_neck) -> (B, 256)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.dim**2)(x).reshape(B, self.dim, self.dim) # (B, 256) -> (B, dim^2) -> (B, dim, dim)
        
        eye = jnp.tile(jnp.eye(self.dim).reshape(1, self.dim, self.dim), (B, 1, 1))
        eye = jax.device_put(eye)
        
        x = x + eye
    
        return x
    
class STNBNReduced(nn.Module):
    dim: int
    stn_bottle_neck: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        B = x.shape[0]
        P = x.shape[1]
        
        x = nn.Conv(features=64, kernel_size=1)(x) # (B, P, dim) -> (B, P, 64)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=128, kernel_size=1)(x) # (B, P, 64) -> (B, P, 128)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=self.stn_bottle_neck, kernel_size=1)(x) # (B, P, 128) -> (B, P, self.stn_bottle_neck)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # x = nn.max_pool(x, window_shape=(P,)).reshape(B, self.stn_bottle_neck) # (B, P, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        x = jnp.max(x, axis=1).reshape(B, self.stn_bottle_neck) # (B, P, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        
        x = nn.Dense(features=self.stn_bottle_neck)(x) # (B, self.stn_bottle_neck) -> (B, self.stn_bottle_neck)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=256)(x) # (B, self.stn_bottle_neck) -> (B, 256)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.dim**2)(x).reshape(B, self.dim, self.dim) # (B, 512) -> (B, dim^2) -> (B, dim, dim)
        
        eye = jnp.tile(jnp.eye(self.dim).reshape(1, self.dim, self.dim), (B, 1, 1))
        eye = jax.device_put(eye)
        
        x = x + eye
    
        return x
    
class PointNetSimplified(nn.Module):
    num_global_feats: int
    hidden_dims: Sequence[int]
    use_layer_norm: bool
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool, encode: bool=False):
        if len(x.shape) == 2:
            x = x[None]
        # Shape of input: (B, P, 3)
        
        # Shape of output: (B, 1024)

        B = x.shape[0]
        P = x.shape[1]
        
        for i, size in enumerate(self.hidden_dims):
            x = nn.Conv(features=size, kernel_size=1, kernel_init=default_init())(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
                
            x = nn.gelu(x)
        
        x = nn.Conv(features=self.num_global_feats, kernel_size=1)(x)
        
        x = jnp.max(x, axis=1).reshape(B, self.num_global_feats) # (B, P, num_global_feats) -> (B, num_global_feats)
        
        return x

class PointNet(nn.Module):
    num_global_feats: int = 256
    stn_bottle_neck: int = 256
    use_second_stn: bool = True
    stn_type: str = "standard"
    use_layernorm: bool = False
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool, encode: bool=False):
        if len(x.shape) == 2:
            x = x[None]
        # Shape of input: (B, P, 3)
        
        # Shape of output: (B, 1024)

        B = x.shape[0]
        P = x.shape[1]
        
        if self.stn_type == "standard":
            trans1 = STN(dim=3, stn_bottle_neck=self.stn_bottle_neck)(x, train=train)
        elif self.stn_type == "bn_reduced":
            trans1 = STNBNReduced(dim=3, stn_bottle_neck=self.stn_bottle_neck)(x, train=train)
        elif self.stn_type == "ln":
            trans1 = STN_LN(dim=3, stn_bottle_neck=self.stn_bottle_neck)(x, train=train)
        else:
            raise NotImplementedError
            
        # print(x.shape, trans1.shape)
        x = jnp.matmul(x, trans1) # (B, P, 3) -> (B, P, 3)
        
        x = nn.Conv(features=64, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 3) -> (B, P, 64)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        else:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # x = nn.Conv(features=64, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 64) -> (B, P, 64)
        # if self.use_layernorm:
        #     x = nn.LayerNorm()(x)
        # else:
        #     x = nn.BatchNorm(use_running_average=not train)(x)
        # x = nn.relu(x)
        
        if self.use_second_stn:
            if self.stn_type == "standard":
                trans2 = STN(dim=64, stn_bottle_neck=self.stn_bottle_neck)(x, train=train)
            elif self.stn_type == "bn_reduced":
                trans2 = STNBNReduced(dim=64, stn_bottle_neck=self.stn_bottle_neck)(x, train=train)
            elif self.stn_type == "ln":
                trans2 = STN_LN(dim=64, stn_bottle_neck=self.stn_bottle_neck)(x, train=train)
            else:
                raise NotImplementedError
            
            x = jnp.matmul(x, trans2) # (B, P, 64) -> (B, P, 64)
        
        # x = nn.Conv(features=64, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 64) -> (B, P, 64)
        # x = nn.relu(x)
        # if self.use_layernorm:
        #     x = nn.LayerNorm()(x)
        # else:
        #     x = nn.BatchNorm(use_running_average=not train)(x)
        
        x = nn.Conv(features=128, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 64) -> (B, P, 128)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        else:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=self.num_global_feats, kernel_size=1, kernel_init=nn.initializers.xavier_normal())(x) # (B, P, 128) -> (B, P, num_global_feats)
        if self.use_layernorm:
            x = nn.LayerNorm()(x)
        else:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # x = nn.max_pool(x, window_shape=(P,)).reshape(B, self.num_global_feats) # (B, P, num_global_feats) -> (B, num_global_feats)
        x = jnp.max(x, axis=1).reshape(B, self.num_global_feats) # (B, P, num_global_feats) -> (B, num_global_feats)
        return x

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, P, C = 20, 500, 3
    pc_key, init_key, dropout_key = jax.random.split(key, 3)
    dummy_pc = jax.random.normal(pc_key, (B, P, C))
    
    stn3 = STN(dim=3, stn_bottle_neck=256)  
    variables = stn3.init(init_key, dummy_pc, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")
    
    out_train, updated = stn3.apply(
        {"params": params, "batch_stats": batch_stats},
        dummy_pc,
        train=True,
        mutable=["batch_stats"],
    )
    
    print(out_train.shape)
    
    pc_key, init_key, dropout_key = jax.random.split(key, 3)
    dummy_pc = jax.random.normal(pc_key, (B, P, C))
    point_net = PointNet(num_global_feats=1024, stn_bottle_neck=256)
    point_net_variables = point_net.init(init_key, dummy_pc, train=True)
    params = point_net_variables['params']
    batch_stats = point_net_variables.get("batch_stats")
    
    out_train, updated = point_net.apply(
        {"params": params, "batch_stats": batch_stats},
        dummy_pc,
        train=True,
        mutable=["batch_stats"],
    )
    
    print(out_train.shape)
      
    # k = 1024
    # stnk = STN(dim=k)