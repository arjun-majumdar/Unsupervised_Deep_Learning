

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Refer-
# https://github.com/facebookresearch/moco/blob/main/moco/builder.py


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op = False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def resnet50_stl10(embed_dim = 128):
    model = torchvision.models.resnet50(weights = None)

    # Change first conv layer of ResNet-50:
    model.conv1 = torch.nn.Conv2d(
        in_channels = 3, out_channels = 64,
        kernel_size = (3, 3), stride = (1, 1),
        padding = (1, 1), bias = False
    )
    model.maxpool = torch.nn.Identity()

    dim_mlp = model.fc.weight.shape[1]

    # dim_mlp
    # 2048

    model.fc = torch.nn.Sequential(
        nn.Linear(in_features = dim_mlp, out_features = dim_mlp, bias = False),
        nn.BatchNorm1d(num_features = dim_mlp),
        nn.ReLU(),
        nn.Linear(in_features = dim_mlp, out_features = embed_dim)
        # model.fc
    )

    return model


def count_trainable_params(model):
    # Count number of layer-wise parameters and total parameters-
    tot_params = 0
    for param in model.parameters():
        # print(f"layer.shape = {param.shape} has {param.nelement()} parameters")
        tot_params += param.nelement()

    return tot_params

# print(f"ResNet-50 CNN + projection head has {count_trainable_params(model)} params")
# ResNet-50 CNN + projection head has 27961024 params


class MoCo(nn.Module):
    """
    Build a MoCo model with:
    1. query encoder
    2. key encoder
    3. queue (dict/list containing negative samples)
    
    Paper: https://arxiv.org/abs/1911.05722
    """
    def __init__(
        self,
        # rank:int,
        embed_dim:int = 128, K:int = 65536,
        m:float = 0.999, T:float = 0.07
    ):
        """
        embed_dim: final embedding feature dimension for loss computation (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum for updating key encoder's paraeters (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.embed_dim = embed_dim

        # Create encoders-
        # Use torchvision ResNet-50 CNN architecture-
        # self.encoder_query = torchvision.models.resnet50(weights = None)
        # self.encoder_key = torchvision.models.resnet50(weights = None)
        self.encoder_query = resnet50_stl10(embed_dim = 128)
        self.encoder_key = resnet50_stl10(embed_dim = 128)

        '''
        # Get shape after avg pool layer-
        dim_mlp = self.encoder_query.fc.weight.shape[1]

        # print(f"ResNet-50 CNN avg-pool output shape: {dim_mlp}")
        # ResNet-50 CNN avg-pool output shape: 2048

        self.encoder_query.fc = nn.Sequential(
            nn.Linear(in_features = dim_mlp, out_features = dim_mlp),
            nn.ReLU(),
            nn.Linear(in_features = dim_mlp, out_features = embed_dim)
        )
        self.encoder_key.fc = nn.Sequential(
            nn.Linear(in_features = dim_mlp, out_features = dim_mlp),
            nn.ReLU(),
            nn.Linear(in_features = dim_mlp, out_features = embed_dim)
        )
        '''

        # I am NOT using shuffling of indices as mentioned in the paper.
        # Instead, I am using Global Batch-norm!
        # Convert BatchNorm to SyncBatchNorm-
        self.encoder_query = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_query)
        self.encoder_key = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_key)

        # Move to GPU-
        # self.encoder_query = self.encoder_query.to(rank)
        # self.encoder_key = self.encoder_key.to(rank)

        
        # Initialize key encoder's parameters using query's parameters
        # and disable gradients for key encoder- 
        for param_q, param_k in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        
        # Create queue-
        '''
        If you have parameters in your model, which should be saved and restored in
        the 'state_dict', but not trained by the optimizer, you should register them
        as buffers.
        Buffers won’t be returned in 'model.parameters()', so that the optimizer won’t
        have a change to update them.

        Refer-
        https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2
        https://stackoverflow.com/questions/57540745/what-is-the-difference-between-register-parameter-and-register-buffer-in-pytorch
        '''
         
        # The idea of a large queue is to decouple it from (GPU) batch-size, and so,
        # it should be decoupled from GPU.
        # All buffers & parameters will be pushed to the device if called
        # on the parent model!
        # self.queue = torch.randn(embed_dim, K)
        # self.queue = nn.functional.normalize(self.queue, dim = 0)
        # self.queue = torch.zeros((embed_dim, K))
        
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype = torch.long))
        # self.queue_ptr = torch.zeros(1, dtype = torch.long)

        self.register_buffer("queue", torch.randn(embed_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim = 0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype = torch.long))

    
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        EWMA update of key encoder's parameters using query encoder's
        parameters updated by SGD.
        """
        for param_q, param_k in zip(
            self.encoder_query.parameters(), self.encoder_key.parameters()
        ):
            # value = (momentum x value) + ((1 - momentum) x current_value)
            param_k.data = (param_k.data * self.m) + (param_q.data * (1.0 - self.m))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # Gather keys (from all GPUs) before updating queue-
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue simultaneously)-
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, imgs_query, imgs_keys):
        """
        Input:
            imgs_query: a batch of query images
            imgs_keys: a batch of key images
        Output:
            logits, targets
        """
        batch_size = imgs_query.shape[0]
        
        # Compute query embeddings/features/encodings-
        query = self.encoder_query(imgs_query)
        query = nn.functional.normalize(query, dim = 1)

        # Compute key embeddings - disable gradients to keys-
        with torch.no_grad():
            # Update key encoder's params using query encoder-
            self._momentum_update_key_encoder()

            keys = self.encoder_key(imgs_keys)

            # L2-normalize output-
            keys = nn.functional.normalize(keys, dim = 1)

            # Ensure: no gradient to keys-
            keys = keys.detach()

        '''
        # WRONG: somehow, torch.bmm() & torch.mm() lead to rising loss! CHECK!!
        # Positive logits (batch_size, 1)-
        logits_pos = torch.bmm(query.view(batch_size, 1, self.embed_dim), keys.view(batch_size, self.embed_dim, 1))

        # Detach 'q' from GPU to CPU-
        logits_neg = torch.mm(query.detach().cpu(), self.queue.clone().detach().cpu())
        # logits_neg = torch.mm(q.view(batch_size, embed_dim).detach().cpu(), queue.view(embed_dim, K))

        # logits batch_size x (1 + K)-
        # logits = torch.cat([logits_pos.view(-1, 1), logits_neg.to(rank)], dim = 1)
        logits = torch.cat([logits_pos.view(-1, 1), logits_neg.cuda()], dim = 1)

        # apply temperature
        logits /= self.T
        '''
        
        # positive logits: Nx1-
        logits_pos = torch.einsum("nc, nc -> n", [query, keys]).unsqueeze(-1)
        # negative logits: NxK-
        logits_neg = torch.einsum("nc, ck -> nk", [query, self.queue.clone().detach()])

        # logits: Nx(1+K)-
        logits = torch.cat([logits_pos, logits_neg], dim = 1)

        # apply temperature-
        logits /= self.T

        # labels: positive key indicators-
        # labels = torch.zeros(logits.shape[0], dtype = torch.long).cuda()
        labels = torch.zeros(logits.shape[0], dtype = torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(keys)

        return logits, labels

