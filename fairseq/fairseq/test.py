import torch
import torch.nn as nn
import os

def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

import copy
import torch.nn.utils.prune as prune
import copy
def get_weight_parameters(layer):
    '''
    Get all parameters/modules identified as 'weight'
    '''
    weight_parameters = []
    if len(list(layer.children())) > 0:
        for child in layer.children():
            for param in child.named_parameters():
                if 'weight' == param[0]:
                    weight_parameters.append((child, param[0]))
            weight_parameters.extend(get_weight_parameters(child))
    
    
    return weight_parameters


def prune_weight_parameters(model, prune_amount):
    '''
    Global pruning
    '''
    params_to_prune = get_weight_parameters(model)
  
    prune.global_unstructured(
        params_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=prune_amount,
    )

    for module, name in params_to_prune:
        try:
            prune.remove(module, name)
            #print(module)
        except Exception as e:
            print(e)
    return model

def get_pruned_models(model, sparsity):
    model_to_prune = copy.deepcopy(model)
    pruned_model = prune_weight_parameters(model, sparsity)
    return pruned_model
	
	
from fairseq.models.bart import BARTModel
bart = BARTModel.from_pretrained('/home/ec2-user/ai-539-nlp-group/bart.base', checkpoint_file='model.pt')
bart.eval()
model = BARTModel.from_pretrained('/home/ec2-user/ai-539-nlp-group/bart.base', checkpoint_file='model.pt')

model.model.load_state_dict(torch.load("/home/ec2-user/ai-539-nlp-group/checkpoints/checkpoint_last.pt"), strict=False)

model.eval()
print(get_model_sparsity(model))