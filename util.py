
import os
from random import sample
from signal import siginterrupt
from typing import Dict, List, Set, Tuple
import numpy as np
import wfdb
import matplotlib.pyplot as plt

import random

def get_record_names(ds_path: str) -> list[str]:
    lines = []
    with open(os.path.join(ds_path, "RECORDS")) as file:
        lines = file.readlines()
        lines = [i.rstrip() for i in lines]
    return lines

"""
    .hea    header file (describes signal file content and format)
    .dat    signal file
    .atr    reference beat annotations (not always available)
    .man    reference beat annotations for selected beats only
    
    Manually determined wave boundaries for selected beats:
    .q1c    2nd pass, annotator 1
    .q2c    2nd pass, annotator 2, available for 11 records only
    .qt1    1st pass, annotator 1
    .qt2    1st pass, annotator 2, available for 11 records only

    Automatically determined wave boundary measurements for all beats:
    .pu     both signals
    .pu0    signal 0 only
    .pu1    signal 1 only
"""

class QT_record():

    name: str
    dat: np.ndarray
    hea: wfdb.Record
    atr: wfdb.Annotation
    q1c: wfdb.Annotation
    man: wfdb.Annotation

    def __init__(self, ds_path: str, record_name: str) -> None:
        file = os.path.join(ds_path, record_name)
        self.name = record_name
        self.hea = wfdb.rdheader(file)
        self.dat = wfdb.rdsamp(file)[0]     # idx [0] of tuple contains the actual data
        #self.atr = wfdb.rdann(file, 'atr')
        self.q1c = wfdb.rdann(file, 'q1c')
        #self.man = wfdb.rdann(file, 'man')
    
    def check_bracket_mismatch(self, fix_mismatches=False) -> Tuple[int, Dict[str, List[int]]]:
        prv = ''
        count = 0
        res: Dict[str, List[int]] = dict()

        new_samples: np.ndarray = self.q1c.sample.copy()
        new_symbols: np.ndarray = self.q1c.symbol.copy()

        for i in range(0, self.q1c.sample.size):

            cur = self.q1c.symbol[i]
            cur_sample = self.q1c.sample[i]
            nxt = self.q1c.symbol[i+1] if i < self.q1c.sample.size - 1 else ''

            if cur not in '()':
                if prv != '(' or nxt != ')':

                    if fix_mismatches:
                        if prv != '(':
                            new_samples = np.insert(new_samples, i + count, cur_sample - 10)
                            new_symbols = np.insert(new_symbols, i + count, '(')
                        else:
                            new_samples = np.insert(new_samples, i + count + 1, cur_sample + 10)
                            new_symbols = np.insert(new_symbols, i + count + 1, ')')

                    count += 1
                    if cur not in res.keys():
                        res[cur] = []
                    res[cur].append(self.q1c.sample[i])

            prv = cur
        
        if fix_mismatches:
            self.q1c.sample = new_samples
            self.q1c.symbol = new_symbols

        b_count, b_count_max = 0, 0
        for i in range(0, self.q1c.sample.size):
            cur = self.q1c.symbol[i]
            if cur == '(':
                b_count += 1
            elif cur == ')':
                b_count -= 1
            b_count_max = max(b_count_max, b_count)

        if fix_mismatches:  # sanity check that all brackets are properly structured
            assert b_count == 0
            assert b_count_max <= 1

        return count, res

    def normalize(self) -> None:
        for i in range(0, self.dat.shape[1]):
            d: np.ndarray = self.dat[:, i].copy()
            d -= np.mean(d)
            d /= np.ptp(d)
            self.dat[:, i] = d

    def plot_ecg(self, track_idx=-1, start=0, stop=0) -> None:

        fig, ax = plt.subplots()
        fig.suptitle('Record {}'.format(self.name))
        fig.set_size_inches(10, 3)
        fig.set_dpi(80)

        avg_sig = np.average(self.dat[:, 0]) - 0.5

        if track_idx == -1:
            for i in range(0, self.dat.shape[1]):
                sig = self.dat[:, i]
                ax.plot(sig, label="track {}".format(i))
        else:
            sig = self.dat[:, track_idx]
            ax.plot(sig, label='track {}'.format(track_idx))

        if stop == 0:
            stop = sig.size
        assert start < stop

        ax.set_xlabel('sample')
        ax.set_ylabel('mV')
        ax.set_xlim(left=start, right=stop)

        for i in range(0, self.q1c.sample.size):
            x, y = self.q1c.sample[i], avg_sig
            marker = self.q1c.symbol[i]
            ax.plot(x, y, marker='${}$'.format(marker), markeredgecolor='red', markerfacecolor='red')

        ax.legend()
        plt.show()

