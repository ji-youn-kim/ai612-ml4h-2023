import torch
import torch.distributed as dist
import distributed.utils as distributed_utils

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        group = distributed_utils.get_data_parallel_group()
        output = distributed_utils.batch_all_gather(input, group=group)

        return tuple(output)
    
    @staticmethod
    def backward(ctx, *grads):
        (input, ) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out