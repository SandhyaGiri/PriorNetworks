import logging
import os
from sacred import Experiment
import numpy as np
from seml import database_utils as db_utils
from seml import misc


ex = Experiment()
misc.setup_logger(ex)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(db_utils.create_mongodb_observer(db_collection, overwrite=overwrite))


@ex.automain
def run(in_domain_dataset, ood_dataset, num_epochs, num_channels, learning_rate, model_dir, data_dir, lr_decay_milestones, train_file, batch_size, logdir):

    logging.info('Received the following configuration:')
    logging.info(f'In domain dataset: {in_domain_dataset}, OOD dataset: {ood_dataset}')
    
    os.system('set | grep SLURM | while read line; do echo "# $line"; done')
    cuda_devices = os.environ['SLURM_JOB_GPUS']
    logging.info(f"GPUs assigned to me: {cuda_devices}")

    lr_decay_milestones = " ".join(map(lambda epoch: "--lrc " + str(epoch), lr_decay_milestones))
    cmd = f'python {train_file} {lr_decay_milestones} --model_dir {model_dir} --normalize --n_channels {num_channels} --batch_size {batch_size} {data_dir} {in_domain_dataset} {ood_dataset} {num_epochs} {learning_rate}'
    logging.info(f"Command being executed: {cmd}")
    os.system(cmd)    

    results = {
        'test_acc': 0.5 + 0.3 * np.random.randn(),
        'test_loss': np.random.uniform(0, 10)
    }
    # the returned result will be written into the database
    return results