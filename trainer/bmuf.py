"""
BMUF (block model update filtering) module
implementation of block model update filtering
"""

import torch
import torch.distributed as dist
#import torch.distributed.ReduceOp as ReduceOp
import torch.nn as nn

SUCCESS = 1
STOP = 0

def _copy_vec_to_param(vec, parameters):
    """Copy vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = param.data.copy_(vec[pointer:pointer + num_param]
                                      .view_as(param).data)
        # Increment the pointer
        pointer += num_param


class BmufTrainer():
    """
    Basic BMUF Trainer Class,
    implements Nesterov Block Momentum

    Args:
        master_node (int): master node index, zero in most cases
        rank (int): local rank, eg, 0-7 if 8GPUs are used
        world_size (int): total number of workers
        model (nn.module): model
        block_momentum (float): block momentum value
        block_lr (float): block learning rate
    """
    def __init__(self, master_node, rank, world_size, model,
                 block_momentum, block_lr):
        self.master_node = master_node
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.block_momentum = block_momentum
        self.block_lr = block_lr
        dist.init_process_group(backend="nccl", init_method="env://")
        #clone() make sure self.param
        #NOT tied to model parameters
        #data() enforces no grad
        param_vec = nn.utils.parameters_to_vector(model.parameters())
        self.param = param_vec.data.clone()
        #broadcast initial param to other nodes
        dist.broadcast(tensor=self.param, src=master_node, async_op=False)
        num_param = self.param.numel()
        if self.rank == master_node:
            self.delta_prev = torch.FloatTensor([0]*num_param).cuda(self.rank)
        else:
            self.delta_prev = None
            #nn.utils.vector_to_parameters(self.param.clone(),
            #                              self.model.parameters())
            _copy_vec_to_param(self.param, self.model.parameters())

    def update_and_sync(self):
        """
        Performs a single block sync and update
        return SUCCESS if numericals are healthy
        return STOP otherwise

        """
        delta = self.param - \
                nn.utils.parameters_to_vector(self.model.parameters()).data
        #gather block gradients into delta
        #default: op=ReduceOp.SUM,
        dist.reduce(tensor=delta, dst=self.master_node)
        #check if model params are still healthy
        if torch.isnan(delta).sum().item():
            return STOP
        if self.rank == self.master_node:
            #for master node
            delta = delta / float(self.world_size)
            self.delta_prev = self.block_momentum * self.delta_prev + \
                              (self.block_lr *(1 - self.block_momentum)* delta)
            self.param -= (1+self.block_momentum) * self.delta_prev
        dist.broadcast(tensor=self.param, src=self.master_node, async_op=False)
        _copy_vec_to_param(self.param, self.model.parameters())

        return SUCCESS

    def broadcast(self, tensor):
        """broadcast interface for trainer"""
        dist.broadcast(tensor=tensor, src=self.master_node, async_op=False)

    def sum_reduce(self, tensor):
        """sumreduce interface for trainer"""
        #op=ReduceOp.SUM,
        dist.reduce(tensor=tensor, dst=self.master_node)


class BlockAdamTrainer():
    """
    This is essentially sync adam optimizer but
    allows each worker to have individual loader
    to improve the training efficiency, to replace
    replace DataParallel()

    Args:
        master_node (int): master node index, zero in most cases
        rank (int): local rank, eg, 0-7 if 8 GPUs are used
        world_size (int): total number of workers
        model (nn.module): torch model
        block_lr (float): block learning rate

    """
    def __init__(self, master_node, rank, world_size, model, block_lr):
        self.master_node = master_node
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.block_lr = block_lr
        dist.init_process_group(backend="nccl", init_method="env://")
        #clone() make sure self.param
        #NOT tied to model parameters
        #data() enforces no grad
        param_vec = nn.utils.parameters_to_vector(model.parameters())
        self.param = nn.parameter.Parameter(param_vec.data.clone())
        #broadcast initial param to other nodes
        dist.broadcast(tensor=self.param.data, src=master_node, async_op=False)
        if self.rank == master_node:
            self.optimizer = torch.optim.Adam([self.param], block_lr, weight_decay=0.0)
        else:
            _copy_vec_to_param(self.param.data, self.model.parameters())

    def update_and_sync(self):
        """Perform a single block sync and update
           when the block size equals to batch size
           we are doing sync adam
        """
        delta = self.param.data - \
                nn.utils.parameters_to_vector(self.model.parameters()).data
        #gather block gradients into delta
        #op=ReduceOp.SUM,
        dist.reduce(tensor=delta, dst=self.master_node)
        #check if model params are still healthy
        if torch.isnan(delta).sum().item():
            return STOP
        if self.rank == self.master_node:
            #local rank is master node
            #delta = delta / float(self.world_size)
            #use delta.data to detach from computation graph
            self.param.grad = delta.data
            self.optimizer.step()
        dist.broadcast(tensor=self.param.data, src=self.master_node, async_op=False)
        _copy_vec_to_param(self.param.data, self.model.parameters())

        return SUCCESS

    def broadcast(self, tensor):
        """broadcast interface for trainer"""
        dist.broadcast(tensor=tensor, src=self.master_node, async_op=False)

    def sum_reduce(self, tensor):
        """sumreduce interface for trainer"""
        #op=ReduceOp.SUM,
        dist.reduce(tensor=tensor, dst=self.master_node)

    def get_block_lr(self):
        """get current learning rate"""
        return self.block_lr

    def set_block_lr(self, value):
        """set a new learning rate"""
        self.block_lr = value
        if self.rank == self.master_node:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value


class BmufAdamTrainer():
    """The implementation of BMUF-adam, check more detils in,
       Chen. et al, 2020, "Parallelizing Adam Optimizer with
       Blockwise Model-Update Filtering."

    Args:
        master_node (int): master node index, zero in most cases
        rank (int): local rank, eg, 0-7 if 8 GPUs are used
        world_size (int): total number of workers
        model (nn.module): torch model
        block_momentum (float): block momentum value
        block_lr (float): block learning rate
        sync_period (int): sync period in number of batches
        optim (torch.optim.Optimizer): adam optimizer
    """
    def __init__(self, master_node, rank, world_size, model,
                 block_momentum, block_lr, sync_period, optim):
        self.master_node = master_node
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.block_momentum = block_momentum
        self.block_lr = block_lr
        self.sync_period = sync_period
        self.optim = optim
        dist.init_process_group(backend="nccl", init_method="env://")
        self.rho = 0.0
        #default setup
        self.betas = (0.9, 0.999)
	#clone() make sure self.param
        #NOT tied to model parameters
        #data() enforces no grad
        param_vec = nn.utils.parameters_to_vector(model.parameters())
        self.param = param_vec.data.clone()
        #broadcast initial param to other nodes
        dist.broadcast(tensor=self.param, src=master_node, async_op=False)
        self.num_param = self.param.numel()
        if self.rank == master_node:
            self.delta_prev = torch.FloatTensor([0]*self.num_param)\
                                   .cuda(master_node)
        else:
            self.delta_prev = None
            _copy_vec_to_param(self.param, self.model.parameters())

        #initialize first and second moment buffer
        dim = 0
        for group in optim.param_groups:
            self.betas = group['betas']
            for p in group['params']:
                dim += p.numel()
        if self.rank == master_node:
            self.exp_avg = torch.FloatTensor([0]*dim).cuda(self.rank)
            self.exp_avg_sq = torch.FloatTensor([0]*dim).cuda(self.rank)
        else:
            self.exp_avg = None
            self.exp_avg_sq = None
        #extend param to accommodate first and second moments
        vec_ext = torch.FloatTensor([0]*dim*2).cuda(self.rank)
        self.param = torch.cat([self.param, vec_ext])

    def update_and_sync(self):
        """perform single block sync and update"""
        #gather block gradients into delta
        delta = self.param[:self.num_param] - \
                nn.utils.parameters_to_vector(self.model.parameters()).data
        #gather local first and second moment
        exp_avg, exp_avg_sq = [], []
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optim.state[p]
                exp_avg.append(state['exp_avg'].view(-1))
                exp_avg_sq.append(state['exp_avg_sq'].view(-1))
        exp_avg = torch.cat(exp_avg)
        exp_avg_sq = torch.cat(exp_avg_sq)
        vec = torch.cat([delta, exp_avg, exp_avg_sq])
        #op=ReduceOp.SUM,
        dist.reduce(tensor=vec, dst=self.master_node)
        #check if model params are still healthy
        if torch.isnan(vec).sum().item():
            return STOP
        self.rho = self.block_momentum * self.rho + self.sync_period
        if self.rank == self.master_node:
            #local rank is master node
            vec = vec / float(self.world_size)
            self.delta_prev = self.block_momentum * self.delta_prev + \
                              (self.block_lr *(1 - self.block_momentum)*\
                               vec[:self.num_param])
            self.param[:self.num_param] -= (1+self.block_momentum) \
                                           * self.delta_prev
            #calculate first and second moment for next block
            dim = (vec.numel() - self.num_param) // 2
            beta1_tau = self.betas[0]**self.sync_period
            beta2_tau = self.betas[1]**self.sync_period
            beta1_rho = self.betas[0]**(self.rho*self.block_momentum)
            beta2_rho = self.betas[1]**(self.rho*self.block_momentum)
            self.exp_avg = beta1_tau * (beta1_rho - 1) * self.exp_avg
            self.exp_avg += (1 - beta1_tau * beta1_rho) *\
                            vec[self.num_param:self.num_param+dim]
            self.exp_avg = self.exp_avg / (1 - beta1_tau)
            self.exp_avg_sq = beta2_tau * (beta2_rho - 1) * self.exp_avg_sq
            self.exp_avg_sq += (1 - beta2_tau * beta2_rho) *\
                               vec[self.num_param+dim:]
            self.exp_avg_sq = self.exp_avg_sq / (1 - beta2_tau)
            self.param[self.num_param:self.num_param+dim] = self.exp_avg
            self.param[self.num_param+dim:] = self.exp_avg_sq

        dist.broadcast(tensor=self.param, src=self.master_node,
                       async_op=False)
        _copy_vec_to_param(self.param[:self.num_param],
                           self.model.parameters())
        #assign flattened moments to optimizer
        ptr1 = self.num_param
        ptr2 = self.num_param+(self.param.numel()-self.num_param)//2
        for group in self.optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optim.state[p]
                state['step'] += self.rho * self.block_momentum
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                numel = exp_avg.numel()
                exp_avg.data = exp_avg.data\
                               .copy_(self.param[ptr1:ptr1+numel]
                                      .view_as(exp_avg).data)
                exp_avg_sq.data = exp_avg_sq.data\
                                  .copy_(self.param[ptr2:ptr2+numel]
                                         .view_as(exp_avg_sq).data)
                ptr1 += numel
                ptr2 += numel


        return SUCCESS

    def broadcast(self, tensor):
        """broadcast interface for trainer"""
        dist.broadcast(tensor=tensor, src=self.master_node, async_op=False)

    def sum_reduce(self, tensor):
        """sum reduce interface for trainer"""
        #op=ReduceOp.SUM,
        dist.reduce(tensor=tensor, dst=self.master_node)
