import torch
import time as time
from utils.utils import set_seed
from utils.logger import Logger


class ExpManager:
    def __init__(self, solver=None):
        self.solver = solver
        self.conf = solver.conf
        self.dataset = solver.dataset
        self.device = torch.device('cuda')
        self.split_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def run(self, n_runs=1):
        assert n_runs <= len(self.split_seeds)
        logger = Logger(
            runs=n_runs,
            ds_name=self.dataset.ds_name,
            calibrator_name = self.conf.calibration["calibrator_name"],
            num_bin=self.conf.calibration["num_bin"],
            dataset=self.dataset,
            conf=self.conf
        )
        succeed = 0
        for i in range(n_runs):
            print("Exp {}/{}".format(i, n_runs))
            set_seed(self.split_seeds[i])

            result = self.solver.run_exp(split=i)
            logger.add_result(succeed, result)

            succeed += 1
            if succeed % n_runs == 0:
                break
        logger.print_statistics()
        # logger.plot()
        logger.save()