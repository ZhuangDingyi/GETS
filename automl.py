search_space = {
    "hidden_dim": {"_type": "choice", "_value": [16, 32, 64]},
    "coef": {"_type": "choice", "_value": [0.1, 0.5, 1.0, 2.0]},
}

from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.experiment_name = 'Calibration'
experiment.config.trial_code_directory = '.'
experiment.config.trial_command = 'python main.py --gpu=3 --n_runs=10 --dataset=photo'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Random'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 100000
experiment.config.trial_concurrency = 1

experiment.run(8093)