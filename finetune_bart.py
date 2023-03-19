from fairseq_cli import train
import torch
from pathlib import Path
import random
import shlex
from contextlib import contextmanager
import sys
import re

@contextmanager
def mock_cli_args(args):
    current_args = sys.argv
    sys.argv = sys.argv[:1] + args
    yield
    sys.argv = current_args


def remove_multiple_whitespaces(text):
    return re.sub(r'  +', ' ', text)

def fairseq_train(
    preprocessed_dir,
    restore_file,
    ngpus=1,
    batch_size=1024,  # Batch size across all gpus (taking update freq into account)
    max_sentences=64,  # Max sentences per GPU
    arch='bart_base',
    save_interval_updates=500,
    max_update=20000,
    total_num_updates=20000,
    lr=3e-05,
    warmup_updates=500,
    lr_scheduler='polynomial_decay',
    criterion='label_smoothed_cross_entropy',
    seed=None,
    fp16=True,
):
    torch.cuda.empty_cache()
    preprocessed_dir = Path(preprocessed_dir)
    total_real_batch_size = max_sentences * ngpus
    update_freq = int(round(batch_size / total_real_batch_size, 0))
    if seed is None:
        seed = random.randint(0, 1000)
    distributed_port = random.randint(10000, 20000)
    args = f'''
    {preprocessed_dir} --restore-file {restore_file} --max-tokens 2048 --task translation 
    --source-lang source --target-lang target --truncate-source
    --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed 
    --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 
    --arch {arch} --criterion {criterion} --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 
    --weight-decay 0.01 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --clip-norm 0.1
    --lr-scheduler {lr_scheduler} --lr {lr} --total-num-update {total_num_updates} --warmup-updates {warmup_updates} --update-freq {update_freq}
    --skip-invalid-size-inputs-valid-test --find-unused-parameters
    --max-update {max_update} --save-interval-updates {save_interval_updates} --keep-interval-updates 1 --patience 10
    --distributed-world-size {ngpus} --distributed-port {distributed_port}
    '''
    if lr_scheduler == 'inverse_sqrt':
        args += '--warmup-init-lr 1e-07'
    if fp16:
        args += f' --fp16'
    args = remove_multiple_whitespaces(args.replace('\n', ' ')).strip(' ')
    # Recover lost quotes around adam betas
    args = re.sub(r'--adam-betas (\(0\.\d+, 0\.\d+\))', r"--adam-betas '\1'", args)
    print(f'fairseq-train {args}')
    with mock_cli_args(shlex.split(args)):
        train.cli_main()

if __name__ == "__main__":
    fairseq_train(preprocessed_dir='cnn_dm-bin', restore_file='bart.base/pruned.pt')
