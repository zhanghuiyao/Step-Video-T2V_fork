import torch.distributed as dist
import xfuser
import torch


# zhy_test: add is_distribute
_is_distribute = False
def is_distribute():
    return _is_distribute


def initialize_parall_group(ring_degree, ulysses_degree):
    dist.init_process_group("nccl")
    xfuser.core.distributed.init_distributed_environment(
        rank=dist.get_rank(), 
        world_size=dist.get_world_size()
    )
    
    xfuser.core.distributed.initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
    )
    torch.cuda.set_device(dist.get_rank())

    global _is_distribute
    _is_distribute = True


def get_parallel_group():
    return xfuser.core.distributed.get_world_group()

def get_sequence_parallel_world_size():
    if is_distribute():
        return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()
    return 1

def get_sequence_parallel_rank():
    if is_distribute():
        return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()
    return 0

def get_sp_group():
    return xfuser.core.distributed.parallel_state.get_sp_group()


def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        output = fn_(_, hidden_states, *args, **kwargs)
        return output
     
    return wrapTheFunction


def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs['parallel']:            
            hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            kwargs['attn_mask'] = torch.chunk(kwargs['attn_mask'], get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        output = fn_(_, hidden_states, *args, **kwargs)
        
        if kwargs['parallel']:
            output = get_sp_group().all_gather(output.contiguous(), dim=-2)
        
        return output
    
    def identityTheFunction(_, hidden_states, *args, **kwargs):
        output = fn_(_, hidden_states, *args, **kwargs)
        return output

    return wrapTheFunction
