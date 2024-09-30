class Recorder:
    def __init__(self, patience=None, criterion=None):
        self.patience = patience
        self.criterion = criterion
        self.best_loss = 1e8
        self.best_metric = -1
        self.wait = 0

    def add(self, metric_val):
        flag = metric_val > self.best_metric

        if flag:
            self.best_metric = metric_val
            self.wait = 0
        else:
            self.wait += 1

        flag_earlystop = self.patience and self.wait >= self.patience

        return flag, flag_earlystop