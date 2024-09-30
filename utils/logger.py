import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

class Logger(object):
    def __init__(self, runs, ds_name, calibrator_name, num_bin, dataset,conf):
        self.ds_name = ds_name
        self.calibrator_name = calibrator_name
        self.num_bin = num_bin
        self.results = [{"uncalibrated": {}, "calibrated": {}} for _ in range(runs)]
        self.dataset = dataset
        self.conf = conf
        self._setup_plot()


    def _setup_plot(self):
        self.figsize = (self.num_bin * 2, 12)
        self.bar_width = 0.5
        self.errwidth = 5
        self.capsize = 8
        self.confidence_color = '#AECDE1'
        self.accuracy_color = '#BBDE93'
        self.diff_color = '#EE9F9B'
        self.root_dir = f'output/{self.calibrator_name}/{self.ds_name}'
        self.font_size = 40
        self.left_adjust = 0.12
        self.right_adjust = 0.95
        self.top_adjust = 0.95
        self.bottom_adjust = 0.24


    def add_result(self, run, result_dict):
        assert run >= 0 and run < len(self.results)
        self.results[run]["uncalibrated"]["acc"] = result_dict["uncalibrated"]["acc"] * 100
        self.results[run]["uncalibrated"]["diff"] = result_dict["uncalibrated"]["diff"] * 100
        self.results[run]["uncalibrated"]["degree_confidence_bined_df"] = result_dict["uncalibrated"]["degree_confidence_bined_df"]
        self.results[run]["uncalibrated"]["degree_accuracy_bined_df"] = result_dict["uncalibrated"]["degree_accuracy_bined_df"]
        self.results[run]["uncalibrated"]["degree_diff_bined_df"] = result_dict["uncalibrated"]["degree_diff_bined_df"]
        self.results[run]['uncalibrated']['index'] = result_dict["uncalibrated"]["index"]
        self.results[run]['uncalibrated']['true'] = result_dict["uncalibrated"]["true"]
        self.results[run]['uncalibrated']['pred_confidence'] = result_dict["uncalibrated"]["pred_confidence"]
        self.results[run]['uncalibrated']['pred'] = result_dict["uncalibrated"]["pred"]
        
        self.results[run]["calibrated"]["acc"] = result_dict["calibrated"]["acc"] * 100
        self.results[run]["calibrated"]["diff"] = result_dict["calibrated"]["diff"] * 100
        self.results[run]["calibrated"]["degree_confidence_bined_df"] = result_dict["calibrated"]["degree_confidence_bined_df"]
        self.results[run]["calibrated"]["degree_accuracy_bined_df"] = result_dict["calibrated"]["degree_accuracy_bined_df"]
        self.results[run]["calibrated"]["degree_diff_bined_df"] = result_dict["calibrated"]["degree_diff_bined_df"]
        self.results[run]["calibrated"]["others"] = result_dict["calibrated"]["others"]
        self.results[run]['calibrated']['index'] = result_dict["calibrated"]["index"]
        self.results[run]['calibrated']['true'] = result_dict["calibrated"]["true"]
        self.results[run]['calibrated']['pred_confidence'] = result_dict["calibrated"]["pred_confidence"]
        self.results[run]['calibrated']['pred'] = result_dict["calibrated"]["pred"]

    def print_statistics(self):
        accs_uncalibrated = torch.tensor([r["uncalibrated"]["acc"] for r in self.results])
        diffs_uncalibrated = torch.tensor([r["uncalibrated"]["diff"] for r in self.results])
        accs_calibrated = torch.tensor([r["calibrated"]["acc"] for r in self.results])
        diffs_calibrated = torch.tensor([r["calibrated"]["diff"] for r in self.results])
        print(f'All runs:')
        print(f'Uncalibrated Test Accuracy: {accs_uncalibrated.mean():.2f} ± {accs_uncalibrated.std():.2f}')
        print(f'Uncalibrated Difference: {diffs_uncalibrated.mean():.2f} ± {diffs_uncalibrated.std():.2f}')
        print(f'Calibrated Test Accuracy: {accs_calibrated.mean():.2f} ± {accs_calibrated.std():.2f}')
        print(f'Calibrated Difference: {diffs_calibrated.mean():.2f} ± {diffs_calibrated.std():.2f}')
        
        import nni
        if nni.get_trial_id()!="STANDALONE":
            metric = {
                'default': float(diffs_calibrated.mean())
            }
            nni.report_final_result(metric)


    def plot(self):
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        uncalibrated_degree_confidence_bined_df = pd.concat([r["uncalibrated"]["degree_confidence_bined_df"] for r in self.results])
        uncalibrated_degree_accuracy_bined_df = pd.concat([r["uncalibrated"]["degree_accuracy_bined_df"] for r in self.results])
        uncalibrated_degree_diff_bined_df = pd.concat([r["uncalibrated"]["degree_diff_bined_df"] for r in self.results])
        calibrated_degree_confidence_bined_df = pd.concat([r["calibrated"]["degree_confidence_bined_df"] for r in self.results])
        calibrated_degree_accuracy_bined_df = pd.concat([r["calibrated"]["degree_accuracy_bined_df"] for r in self.results])
        calibrated_degree_diff_bined_df = pd.concat([r["calibrated"]["degree_diff_bined_df"] for r in self.results])
        self._plot_combined_df(uncalibrated_degree_confidence_bined_df, f'{self.root_dir}/confidence/uncalibrated_confidence.png', self.confidence_color, 'confidence')
        self._plot_combined_df(uncalibrated_degree_accuracy_bined_df, f'{self.root_dir}/accuracy/uncalibrated_accuracy.png', self.accuracy_color, 'accuracy')
        self._plot_combined_df(uncalibrated_degree_diff_bined_df, f'{self.root_dir}/diff/uncalibrated_diff.png', self.diff_color, 'difference')
        self._plot_combined_df(calibrated_degree_confidence_bined_df, f'{self.root_dir}/confidence/calibrated_confidence.png', self.confidence_color, 'confidence')
        self._plot_combined_df(calibrated_degree_accuracy_bined_df, f'{self.root_dir}/accuracy/calibrated_accuracy.png', self.accuracy_color, 'accuracy')
        self._plot_combined_df(calibrated_degree_diff_bined_df, f'{self.root_dir}/diff/calibrated_diff.png', self.diff_color, 'difference')
    
    def save(self):
        
        if self.conf.calibration['calibrator_name']=='GETS':            
            backbone = self.conf.calibration['backbone']
            torch.save(self.results,f'{self.root_dir}/{backbone}_results.pt')
        else:
            torch.save(self.results, f'{self.root_dir}/results.pt')            
        if "node_gates" in self.results[0]["calibrated"]["others"]:
            torch.save(self.results[0]["calibrated"]["others"]["node_gates"], f'{self.root_dir}/{backbone}_node_gates.pt')
            test_idx = self.dataset.test_idxs[0]
            torch.save(test_idx, f'{self.root_dir}/{backbone}_test_idx.pt')
        

    def _plot_combined_df(self, df, path, color, y_label):
        df_mean = df.groupby('degree_range', sort=False).mean().reset_index()
        df_std = df.groupby('degree_range', sort=False).std().reset_index()
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel('degree range', fontsize=self.font_size)
        ax.set_ylabel(y_label, fontsize=self.font_size)
        if y_label != "difference":
            y = (y_label, 'mean')
        else:
            y = y_label
        sns.barplot(
            x='degree_range',
            y=y,
            data=df_mean,
            color=color,
            ax=ax,
            width=self.bar_width
        )

        # x = np.arange(len(df_mean['degree_range']))
        # ax.errorbar(
        #     x, df_mean[y], yerr=df_std[y_label].values.flatten(),
        #     fmt='none', ecolor='black', elinewidth=self.errwidth, capsize=self.capsize, capthick=self.errwidth
        # )

        ax.tick_params(axis='y')
        plt.xticks(rotation=30, fontsize=self.font_size)
        plt.yticks(fontsize=self.font_size)
        plt.grid(True)
        plt.subplots_adjust(
            left=self.left_adjust, 
            right=self.right_adjust, 
            top=self.top_adjust, 
            bottom=self.bottom_adjust
        )
        plt.savefig(path)