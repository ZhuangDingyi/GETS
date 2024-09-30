from model.gnns import load_gnn
from utils.recorder import Recorder
from utils.utils import accuracy, setup_directories
import torch
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from model.calibrator import TS, ETS, VS, CaGCN, GATS, CaGCN_MoE

class Solver:
    def __init__(self, conf, dataset):
        self.dataset = dataset
        self.conf = conf
        self.device = self.dataset.device
        self.calibrator_name = self.conf.calibration['calibrator_name']
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        in_dim = self.dataset.features.shape[1]
        out_dim = self.dataset.num_classes
        self.conf.gnn["in_dim"] = in_dim
        self.conf.gnn["out_dim"] = out_dim
        self.num_bin = self.conf.calibration['num_bin']
        try:
            setup_directories('output', self.calibrator_name, dataset.ds_name)
        except:
            pass
    
    def run_exp(self, split=0):
        self._set(split)
        print("************************************")
        print("Start fitting model")
        print("************************************")
        self._learn()
        print("************************************")
        print("Start fitting calibration")
        print("************************************")
        print("Calibration model configuration")
        print(self.conf)
        print("************************************")
        self._calibrate()
        print("************************************")
        print("GPU memory allowcation")
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2  # Memory allocated by tensors
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2  # Memory reserved by the allocator
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
        print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")
        return self.result
    
    
    def _setup_result_dict(self):
        dict_template = {
            "acc": None,
            "diff": None,
            "degree_confidence_bined_df": None,
            "degree_accuracy_bined_df": None,
            "degree_diff_bined_df": None,
            
            "index": None,
            "true": None,
            "pred": None,
            "pred_confidence": None
        }
        
        self.result = {
            "uncalibrated": dict_template.copy(),
            "calibrated": dict_template.copy()
        }

    def _set(self, run):
        self.model = load_gnn(self.conf).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.train["lr"], weight_decay=self.conf.train["weight_decay"])
        self.best_val_acc = -1
        self.recorder = Recorder(self.conf.train['patience'])
        self.weights = None
        self._setup_result_dict()
        self.train_idx = self.dataset.train_idxs[run]
        self.val_idx = self.dataset.val_idxs[run]
        self.test_idx = self.dataset.test_idxs[run]

    
    def _learn(self):
        for epoch in range(self.conf.train["epochs"]):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(self.dataset.g, self.dataset.features)
            loss = self.loss_fcn(logits[self.train_idx], self.dataset.labels[self.train_idx])
            loss.backward()
            self.optimizer.step()
            acc_train = accuracy(logits[self.train_idx], self.dataset.labels[self.train_idx])
            acc_val = self._evaluate(mode='val')
            flag, flag_earlystop = self.recorder.add(acc_val)
            if flag:
                self.weights = self.model.state_dict()
            if flag_earlystop:
                print("Early stopping at epoch {}".format(epoch))
                break
            print("Epoch {:05d} | Loss(train) {:.4f} | Acc(train) {:.4f} | Acc(val) {:.4f} |{}"
                  .format(epoch + 1, loss.item(), acc_train, acc_val, "*" if flag else ""))
        self.model.load_state_dict(self.weights)
        
        self.result['uncalibrated']['index'] = self.test_idx        
        self.result['uncalibrated']['true'] = self.dataset.labels[self.test_idx].cpu().numpy()
        self.result['uncalibrated']['pred'],self.result['uncalibrated']['pred_confidence'] = self._save_nodewise_results(mode='test')

        acc, diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df, others = self._test()
        self.result['uncalibrated']['acc'] = acc
        self.result['uncalibrated']['diff'] = diff
        self.result['uncalibrated']['degree_confidence_bined_df'] = degree_confidence_bined_df
        self.result['uncalibrated']['degree_accuracy_bined_df'] = degree_accuracy_bined_df
        self.result['uncalibrated']['degree_diff_bined_df'] = degree_diff_bined_df
        self.result['uncalibrated']['others'] = others

    def _save_nodewise_results(self, mode):
        if mode == 'val':
            idx = self.val_idx
            model = self.model
        elif mode == 'test':
            idx = self.test_idx
            model = self.model
        elif mode == 'calibration':
            idx = self.test_idx
            model = self.calibrated_model
        
        model.eval()
        with torch.no_grad():
            if self.calibrator_name == 'MoE' and mode == 'calibration':
                logits, _, node_gates = model(self.dataset.g, self.dataset.features)                
            else:
                logits = model(self.dataset.g, self.dataset.features)
        if mode in ['test', 'calibration']:
            softmax_values = torch.softmax(logits, dim=1).cpu().numpy()
            confidence = np.amax(softmax_values, axis=1)            
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            return pred,confidence
        else:
            return None
        
    
    def _evaluate(self, mode):
        if mode == 'val':
            idx = self.val_idx
            model = self.model
        elif mode == 'test':
            idx = self.test_idx
            model = self.model
        elif mode == 'calibration':
            idx = self.test_idx
            model = self.calibrated_model
        others = {}
        model.eval()
        with torch.no_grad():
            if self.calibrator_name == 'MoE' and mode == 'calibration':
                logits, _, node_gates = model(self.dataset.g, self.dataset.features)
                others['node_gates'] = node_gates
            else:
                logits = model(self.dataset.g, self.dataset.features)
            acc = accuracy(logits[idx], self.dataset.labels[idx])
        if mode == 'val':
            return acc
        elif mode in ['test', 'calibration']:
            diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df  = self._get_diff(logits[idx], self.dataset.labels[idx])
            return acc, diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df, others
        
        
    def _test(self):
        acc, diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df, others = self._evaluate(mode='test')
        return acc, diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df, others
    
    def _get_diff(self, logits, labels):
        # Calculate degree confidence dataframe
        softmax_values = torch.softmax(logits, dim=1).cpu().numpy()
        confidence_values = np.amax(softmax_values, axis=1)
        degrees = self.dataset.g.in_degrees()[self.test_idx].cpu().numpy()
        degree_confidence_df = pd.DataFrame({
            'degree': degrees, 
            'confidence': confidence_values
        })
        degree_confidence_df = degree_confidence_df.sort_values(by='degree').reset_index()
        self.bin_size = math.ceil(len(degree_confidence_df) / self.num_bin)
        degree_confidence_df['bin'] = degree_confidence_df.index // self.bin_size
        degree_confidence_bined_df = degree_confidence_df.groupby('bin').agg({
            'degree': ['min', 'max'],
            'confidence': 'mean'
        }).reset_index()
        self.bin_connts = degree_confidence_df.groupby('bin').size().values
        degree_confidence_bined_df['count'] = self.bin_connts
        self.degree_range = degree_confidence_bined_df.apply(
            lambda x: f"[{int(x[('degree', 'min')])}, {int(x[('degree', 'max')])}]", axis=1)
        degree_confidence_bined_df['degree_range'] = self.degree_range

        # Calculate degree accuracy dataframe
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        correct_predictions = predictions == labels.cpu().numpy().astype(int)
        degree_accuracy_df = pd.DataFrame({
            'degree': degrees,
            'accuracy': correct_predictions
        })
        degree_accuracy_df = degree_accuracy_df.sort_values(by='degree').reset_index()
        degree_accuracy_df['bin'] = degree_accuracy_df.index // self.bin_size
        degree_accuracy_bined_df = degree_accuracy_df.groupby('bin').agg({
            'degree': ['min', 'max'],
            'accuracy': 'mean'
        }).reset_index()
        degree_accuracy_bined_df['count'] = self.bin_connts
        degree_accuracy_bined_df['degree_range'] = self.degree_range
        degree_accuracy_bined_df = degree_accuracy_bined_df
        
        # Calculate confidence accuracy difference
        diff = np.abs(
            degree_confidence_bined_df['confidence']['mean'] - degree_accuracy_bined_df['accuracy']['mean']
        )
        self.degree_weights = degree_confidence_bined_df.apply(lambda x: x[('count', '')] / len(degree_confidence_df), axis=1)
        weighted_sum_diff = np.sum(diff * self.degree_weights)

        degree_diff_bined_df = pd.DataFrame({
            'degree_range': self.degree_range,
            'difference': diff,
        })

        return weighted_sum_diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df

    def _calibrate(self):
        if self.calibrator_name == 'MoE':
            self.calibrated_model = CaGCN_MoE(
                    self.model,
                    self.dataset.features.shape[1],
                    self.dataset.num_classes,
                    self.device,
                    self.conf
                )
            self.calibrated_model.fit(
                self.dataset.g,
                self.dataset.features,
                self.dataset.labels,
                [self.train_idx, self.val_idx, self.test_idx]
            )
        elif self.calibrator_name == 'VS':
            self.calibrated_model = VS(
                self.model,
                self.dataset.num_classes,
                self.device,
                self.conf
            )
            self.calibrated_model.fit(
                self.dataset.g,
                self.dataset.features,
                self.dataset.labels,
                [self.train_idx, self.val_idx, self.test_idx]
            )
        elif self.calibrator_name == 'TS':
            self.calibrated_model = TS(
                self.model,
                self.device,
                self.conf
            )
            self.calibrated_model.fit(
                self.dataset.g,
                self.dataset.features,
                self.dataset.labels,
                [self.train_idx, self.val_idx, self.test_idx]
            )
        elif self.calibrator_name == 'ETS':
            self.calibrated_model = ETS(
                self.model,
                self.dataset.num_classes,
                self.device,
                self.conf
            )
            self.calibrated_model.fit(
                self.dataset.g,
                self.dataset.features,
                self.dataset.labels,
                [self.train_idx, self.val_idx, self.test_idx]
            )
        elif self.calibrator_name == 'CaGCN':
            self.calibrated_model = CaGCN(
                self.model,
                self.dataset.num_classes,
                self.device,
                self.conf
            )
            self.calibrated_model.fit(
                self.dataset.g,
                self.dataset.features,
                self.dataset.labels,
                [self.train_idx, self.val_idx, self.test_idx]
            )
        elif self.calibrator_name == 'GATS':
            self.calibrated_model = GATS(
                self.model,
                self.dataset.g,
                self.dataset.num_classes,
                self.train_idx,
                self.device,
                self.conf
            )
            self.calibrated_model.fit(
                self.dataset.g,
                self.dataset.features,
                self.dataset.labels,
                [self.train_idx, self.val_idx, self.test_idx]
            )

        
        self.result['calibrated']['index'] = self.test_idx
        assert (self.result['calibrated']['index'] == self.result['uncalibrated']['index']).all()
        
        self.result['calibrated']['true'] = self.dataset.labels[self.test_idx].cpu().numpy()
        self.result['calibrated']['pred'],self.result['calibrated']['pred_confidence'] = self._save_nodewise_results(mode='calibration')
        
        acc, diff, degree_confidence_bined_df, degree_accuracy_bined_df, degree_diff_bined_df, others = self._evaluate(mode='calibration')
        self.result['calibrated']['acc'] = acc
        self.result['calibrated']['diff'] = diff
        self.result['calibrated']['degree_confidence_bined_df'] = degree_confidence_bined_df
        self.result['calibrated']['degree_accuracy_bined_df'] = degree_accuracy_bined_df
        self.result['calibrated']['degree_diff_bined_df'] = degree_diff_bined_df
        self.result['calibrated']['others'] = others