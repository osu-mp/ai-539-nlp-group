import torch
import torch.nn as nn
from fairseq.models.bart import BARTModel
from tqdm import tqdm


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

def get_pruned_state(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith("_orig"):
            new_state_dict[k[:-5]] = v * state_dict[k[:-5] + "_mask"] 
        elif k.endswith("_mask"):
            pass
        else:
            new_state_dict[k] = v
    return new_state_dict

@torch.no_grad()
def generate(bart, infile, outfile="bart_hypo.txt", bsz=32, n_obs=None, **eval_kwargs):
    count = 1

    with open(infile) as source, open(outfile, "w") as fout:
        all_lines = source.readlines()
        slines = []
        for sline in tqdm(all_lines):
            if n_obs is not None and count > n_obs:
                break
            if count % bsz == 0:
                hypotheses_batch = bart.sample(slines, **eval_kwargs)
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1

        if slines != []:
            hypotheses_batch = bart.sample(slines, **eval_kwargs)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()

if __name__ == "__main__":
    print("Loading bart base original")
    bart = BARTModel.from_pretrained('./path_to/bart.base', checkpoint_file='model.pt')
    print("Loading bart base modified")
    bart_modified = torch.load('path_to/bart.base/modified.pt')
    print("Load modified state into original")
    bart.model.load_state_dict(get_pruned_state(bart_modified['model']))
    print(f"Sparsity: {get_model_sparsity(bart.model)}")
    CNN_KWARGS = dict(beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
    infile = './path_to/cnn_dm/test.source'
    outfile = './path_to/cnn_dm/test_eval.hypo'

    eval_kwargs = CNN_KWARGS
    bart = bart.eval()
    if torch.cuda.is_available():
        bart = bart.cuda().half()
    generate(
        bart, infile, outfile, **eval_kwargs
    )