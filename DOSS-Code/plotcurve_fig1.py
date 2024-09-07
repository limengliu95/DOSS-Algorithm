#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import glob, os, math, sys
from itertools import chain
import xlsxwriter


# plt.style.use('ggplot')

class TestResult:
    def __init__(self, prob, prob_dim, strgy,
                 max_evals, num_trial, num_init, fX, X=None):
        self.prob = prob
        self.prob_dim = prob_dim
        self.strgy = strgy
        self.max_evals = max_evals
        self.num_trial = num_trial
        self.num_init = num_init
        self.X = X
        # print(fX.shape)
        if fX.shape[1] >= max_evals:
            self.fX = fX[0:max_evals]
        else:
            self.fX = np.hstack((fX, np.min(fX) * np.ones((1, max_evals - fX.shape[1]))))

    def addTrials(self, fX):
        if self.num_trial < 30:
            self.fX = np.vstack((self.fX, fX[0:self.max_evals]))
            self.num_trial += fX.shape[0]


def readResults(dir_result, set_prob, set_strategy, dict_probstrgy, result):
    # Read files and save results
    for f in np.sort(os.listdir(dir_result)):
        if f.endswith('.npz') == False:
            continue
        # print(f)
        data = np.load(dir_result + '/' + f, allow_pickle=True)
        r_prob = str(data['prob'])
        r_strgy = data['strgy']
        if ("_std" in dir_result):
            r_strgy = ["SDSGDYCK_std"]
        r_prob_dim = data['prob_dim']
        r_max_evals = data['max_evals']
        r_num_trial = data['num_trial']
        r_num_init = data['num_init']
        if 'result' in data.files:
            r_fX = data['result']
        # r_num_tr = data['num_tr']
        elif 'result_fX' in data.files:
            r_fX = data['result_fX']

        if 'result_X' in data.files:
            r_X = data['result_X']
        else:
            r_X = []

        if r_num_init == 0:
            if r_strgy == 'DYCORS':
                r_num_init = 2 * r_prob_dim + 2
            elif r_strgy == 'DS' or r_strgy == 'RBFOpt':
                r_num_init = round(0.5 * (r_prob_dim + 1)) if r_prob_dim < 20 else round(0.4 * (r_prob_dim + 1))

        # print(r_prob)
        # print(r_strgy)
        # print(r_prob_dim)
        # print(r_max_evals)
        # print(r_num_trial)
        # print(r_result)
        if r_prob == 'Camel' or r_prob == 'Shekel5' or r_prob == 'Shekel7' or r_prob == 'Shekel10':
            continue

        if r_prob + str(r_prob_dim) not in set_prob:
            set_prob.add(r_prob + str(r_prob_dim))
        for strgy in r_strgy:
            if strgy not in set_strategy:
                set_strategy.append(strgy)

        num_method = 0
        r_num_tr = [1, 20]
        for i in range(len(r_fX)):
            name_strgy = r_strgy[num_method]
            if name_strgy == 'TuRBO':
                name_strgy = name_strgy + '_' + str(r_num_tr[i - num_method])
                if i - num_method == len(r_num_tr) - 1:
                    num_method += 1
            else:
                num_method += 1

            # if name_strgy == 'CK':
            # continue
            # if name_strgy == 'MC':
            # continue

            # if r_X:
            #     test_result = TestResult(prob=r_prob, prob_dim=r_prob_dim,
            #         strgy=name_strgy, max_evals=r_max_evals, num_trial=r_num_trial,
            #         fX=r_fX[i], X=r_X[i])
            # else:
            test_result = TestResult(prob=r_prob, prob_dim=r_prob_dim,
                                     strgy=name_strgy, max_evals=r_max_evals, num_trial=r_num_trial,
                                     num_init=r_num_init, fX=r_fX[i])
            if dict_probstrgy.get(r_prob + str(r_prob_dim) + name_strgy) != None:
                # print('New Result Found for Problem: '+r_prob+str(r_prob_dim)+', Strategy: '+name_strgy)
                if r_max_evals != result[dict_probstrgy[r_prob + str(r_prob_dim) + name_strgy]].max_evals:
                    result[dict_probstrgy[r_prob + str(r_prob_dim) + name_strgy]] = test_result
                else:
                    # print(r_prob)
                    # print(test_result.fX.shape)
                    result[dict_probstrgy[r_prob + str(r_prob_dim) + name_strgy]].addTrials(test_result.fX)
            else:
                # print('Add Result for Problem: '+r_prob+str(r_prob_dim)+', Strategy: '+name_strgy)
                dict_probstrgy[r_prob + str(r_prob_dim) + name_strgy] = len(result)
                result.append(test_result)


def readFromFolders(subset):
    readResults("./results/SDSGDYCK/" + subset, set_prob, set_strategy, dict_probstrgy, result)
    #readResults("./results/CK/"+subset, set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/SDSG/"+subset, set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/SDSGCK/"+subset, set_prob, set_strategy, dict_probstrgy, result)
    readResults("./results/SDSGDYCK_std/"+subset, set_prob, set_strategy, dict_probstrgy, result)
    readResults("./results/DYCORS/" + subset, set_prob, set_strategy, dict_probstrgy, result)
    readResults("./results/TuRBO/" + subset, set_prob, set_strategy, dict_probstrgy, result)
    readResults("./results/RBFOpt/" + subset, set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/SDSGCK_SLHD/"+subset, set_prob, set_strategy, dict_probstrgy, result)
    #readResults("./results/MADS/" + subset, set_prob, set_strategy, dict_probstrgy, result)
    #readResults("./results/GA/" + subset, set_prob, set_strategy, dict_probstrgy, result)
    #readResults("./results/CMA-ES/" + subset, set_prob, set_strategy, dict_probstrgy, result)


def readTestset1():
    #readFromFolders("EASY6")
    readFromFolders("EASY36")
    readFromFolders("EASY48")
    readFromFolders("EASY60")
    readFromFolders("NEW36")
    readFromFolders("NEW48")
    readFromFolders("NEW60")


tdist95 = [12.69, 4.271, 3.179, 2.776, 2.570, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
           2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
           2.021, 2.009, 2.000, 1.994, 1.990, 1.987, 1.984]
# colormarker = ['b-v', 'r-^', 'g-o', 'k-*', 'c->', '-<', '-s']
# color = ['b', 'r', 'g', 'k', 'c']

marker = ['-v', '-^', '-o', '-*', '->', '-<', '-s', '-d', '-p', '-h']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9', 'C7']

# dir_result = './results/0703_0/AZRSLGWRM50'
# dir_result = './results/0703_0/AZRSLGWRM100'
# dir_result = './results/0703_0/BBOB40'
# dir_result = './results/0718_1/BBOB40'
# dir_result = './results/0719_0/EASY100'

set_prob = set()
set_strategy = list()
dict_probstrgy = {}
result = []

# dir_result = './results/' + str(input("results dir: "))
dir_result = './results/' + 'testset1'
if dir_result != './results':
    if "EASY30" in dir_result:
        readResults("./results/DYCORS/EASY/30", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/TuRBO/EASY/30", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/RBFOpt/EASY/30", set_prob, set_strategy, dict_probstrgy, result)
    elif "EASY50" in dir_result:
        # readResults("./results/DYCORS/EASY/50", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/TuRBO/EASY/50", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/RBFOpt/EASY/50", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/0913_1/EASY50/DYCORS", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/0913_1/EASY50/RBFOpt", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/0913_1/EASY50/DS", set_prob, set_strategy, dict_probstrgy, result)
    elif "EASY100" in dir_result:
        readResults("./results/DYCORS/EASY100", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/RBFOpt/EASY100", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/DYCORS/EASY/100", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/RBFOpt/EASY/100", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0802_0/EASY100/DS0", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0802_0/EASY100/DS1", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0802_0/EASY100/DS2", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0802_0/EASY100/DS3", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0802_0/EASY100/DS4", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0802_0/EASY100/DS5", set_prob, set_strategy, dict_probstrgy, result)
    # readResults("./results/0801_0/EASY100/CK", set_prob, set_strategy, dict_probstrgy, result)
    elif "EASY200" in dir_result:
        readResults("./results/DYCORS/EASY/200", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/TuRBO/EASY/200/201i", set_prob, set_strategy, dict_probstrgy, result)
    elif "BBOB40" in dir_result:
        readResults("./results/DYCORS/BBOB/40", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/TuRBO/BBOB/40", set_prob, set_strategy, dict_probstrgy, result)
    elif "BBOB50" in dir_result:
        readResults("./results/DYCORS/BBOB/50", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/TuRBO/BBOB/50", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/RBFOpt/BBOB/50", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/0913_1/BBOB50/DYCORS", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/0913_1/BBOB50/RBFOpt", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/0913_1/BBOB50/DS", set_prob, set_strategy, dict_probstrgy, result)
    elif "BBOB80" in dir_result:
        readResults("./results/DYCORS/BBOB/80", set_prob, set_strategy, dict_probstrgy, result)
    elif "BBOB100" in dir_result:
        readResults("./results/DYCORS/BBOB/100", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/RBFOpt/BBOB/100", set_prob, set_strategy, dict_probstrgy, result)
        # readResults("./results/TuRBO/BBOB/100", set_prob, set_strategy, dict_probstrgy, result)
    elif "BBOB160" in dir_result:
        readResults("./results/DYCORS/BBOB/160", set_prob, set_strategy, dict_probstrgy, result)
    # elif "ML" in dir_result:
    #     readResults("./results/DYCORS/ML", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/TuRBO/ML", set_prob, set_strategy, dict_probstrgy, result)
    # elif "NEW" in dir_result:
    #     readResults("./results/1106_0/NEW/DYCORS", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/1106_0/NEW/RBFOpt", set_prob, set_strategy, dict_probstrgy, result)
    # elif "EASY60" in dir_result:
    #     readResults("./results/DYCORS/EASY60", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/RBFOpt/EASY60", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/TuRBO/EASY60", set_prob, set_strategy, dict_probstrgy, result)
    # elif "NEW60" in dir_result:
    #     readResults("./results/DYCORS/NEW60", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/RBFOpt/NEW60", set_prob, set_strategy, dict_probstrgy, result)
    elif "BBOB60" in dir_result:
        readResults("./results/DYCORS/BBOB60", set_prob, set_strategy, dict_probstrgy, result)
        readResults("./results/RBFOpt/BBOB60", set_prob, set_strategy, dict_probstrgy, result)
    # elif "Rover" in dir_result:
    #     readResults("./results/DYCORS/ML/Rover", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/RBFOpt/ML/Rover", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/TuRBO/ML/Rover", set_prob, set_strategy, dict_probstrgy, result)
    elif "EASY36" in dir_result:
        readFromFolders("EASY36")
    elif "EASY48" in dir_result:
        readFromFolders("EASY48")
    elif "EASY60" in dir_result:
        readFromFolders("EASY60")
    elif "NEW36" in dir_result:
        readFromFolders("NEW36")
    elif "NEW48" in dir_result:
        readFromFolders("NEW48")
    elif "NEW60" in dir_result:
        readFromFolders("NEW60")
    elif "BBOB36" in dir_result:
        readFromFolders("BBOB36")
    elif "Rover3000" in dir_result:
        readFromFolders("Rover3000")
        readResults("./results/TuRBO/Rover3000", set_prob, set_strategy, dict_probstrgy, result)
    # elif "testset1" in dir_result:
    #     readResults("./results/DYCORS/testset1", set_prob, set_strategy, dict_probstrgy, result)
    #     readResults("./results/RBFOpt/testset1", set_prob, set_strategy, dict_probstrgy, result)
    elif "testset1" in dir_result:
        readTestset1()

    readResults(dir_result, set_prob, set_strategy, dict_probstrgy, result)

# Plotting

nprob = len(set_prob)
# nrow = int(input("Number of rows: "))
# ncol = int(input("Number of columns: "))
nrow = 6
ncol = 7
print("Number of Problem: %d" % nprob)
ifig = 0
iplot = 0
evals_per_fig = 20000
nsubfig_per_prob = 1

if nrow == 2 and ncol == 5:
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.98, hspace=0.3, wspace=0.4)
elif nrow == 3 and ncol == 3:
    fig = plt.figure(figsize=(7.5, 7.5))
    plt.subplots_adjust(top=0.97, bottom=0.06, left=0.08, right=0.98, hspace=0.35, wspace=0.4)
else:
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.05, right=0.98, hspace=0.3, wspace=0.4)

current_prob = ' '
# truncation = int(input("Truncation: "))
truncation = 50000
fval_best = float('inf')
dist_best_min = float('inf')
set_prob = list(set_prob)
set_prob.sort()
xlsx_col_prob = []
xlsx_col_fval = []
xlsx_col_qap = []
dict_fval_best = {}
for prob in set_prob:
    # print("Prob: "+prob)
    ifig += 1
    ax = fig.add_subplot(nrow, ncol * nsubfig_per_prob, ifig)
    plt.grid()
    fval_best = float('inf')
    dist_best_min = float('inf')
    ylim_ub = -float('inf')
    ylim_ratio = 10
    iplot = 0
    xlsx_row = []
    for test_result in result:
        # print("Result: "+test_result.prob+str(test_result.prob_dim))
        if prob != test_result.prob + str(test_result.prob_dim):
            continue
            # if ifig != 0:
            # print(ax.get_ylim())
            # ax.set_ylim(ax.get_ylim()[0], min(ax.get_ylim()[1], ax.get_ylim()[0]+ylim_ratio*dist_best_min))
        # prob = test_result.prob
        current_prob = test_result.prob
        prob_dim = test_result.prob_dim
        strgy = test_result.strgy
        max_evals = test_result.max_evals
        num_trial = test_result.num_trial
        num_init = test_result.num_init
        if num_init == 0:
            if strgy == 'DYCORS':
                num_init = 2 * prob_dim + 2
            elif strgy == 'CK' or strgy == 'SDSG' or strgy == 'SDSGCK' or strgy == 'SDSGDY' or \
                    strgy == 'SDSGCK' or strgy == 'SDSGDYCK' or strgy == 'RBFOpt':
                num_init = round(0.5 * (prob_dim + 1)) if prob_dim < 20 else round(0.4 * (prob_dim + 1))
        # print(num_init)
        fX = test_result.fX
        X = test_result.X

        fX = np.minimum.accumulate(fX, axis=1)
        result_ave = np.mean(fX, axis=0)[0:truncation]
        if num_trial == 1:
            error_bar = np.zeros((truncation,))
        elif num_trial <= 30:
            result_std = np.std(fX, axis=0)[0:truncation]
            error_bar = result_std / np.sqrt(num_trial)
            error_bar *= tdist95[num_trial - 2]
        # elif num_trial == 40:
        else:
            error_bar = np.zeros((truncation,))

        # result_wst = np.max(fX, axis=0)[0:truncation]
        # result_bst = np.min(fX, axis=0)[0:truncation]

        ylim_ub = max(ylim_ub, result_ave[2 * prob_dim + 1])
        fval = np.min(result_ave, axis=0)
        xlsx_row.append(fval)
        if fval < fval_best:
            if fval_best != float('inf'):
                dist_best_min = min(fval_best - fval, dist_best_min)
            fval_best = fval

        if fval != fval_best and fval - fval_best < dist_best_min:
            dist_best_min = fval - fval_best
        # print(strgy+': %.3f, Current Best: %.3f, Current Minimal Distance: %.3f' % (fval, fval_best, dist_best_min))
        iter = np.array(list(range(1, result_ave.shape[0] + 1)))
        if prob_dim == 200:
            start = 400
        else:
            start = 0
        end = min(truncation, max_evals)

        if strgy == 'DYCORS':
            mk = marker[0]
            color = colors[0]
        elif strgy == 'TuRBO_1':
            mk = marker[1]
            color = colors[1]
        elif strgy == 'RBFOpt':
            mk = marker[2]
            color = colors[2]
        elif strgy == 'SDSGDYCK':
            mk = marker[3]
            color = colors[3]
        elif strgy == 'CK':
            mk = marker[4]
            color = colors[4]
        elif strgy == 'SDSG':
            mk = marker[5]
            color = colors[5]
            strgy = "SG"
        elif strgy == 'SDSGCK':
            mk = marker[6]
            color = colors[6]
        elif strgy == 'SDSGDY':
            mk = marker[7]
            color = colors[7]
        elif strgy == 'SDSGDYCK_std':
            mk = marker[7]
            color = colors[7]
            strgy = "DOSS(std)"
        elif strgy == 'SDSGCK_SLHD':
            mk = marker[8]
            color = colors[8]
        elif strgy == 'GA':
            mk = marker[4]
            color = colors[4]
            strgy = "GA"
        elif strgy == 'CMA-ES':
            mk = marker[5]
            color = colors[5]
            strgy = "CMA-ES"

        elif strgy == 'MADS':
            mk = marker[9]
            color = colors[9]

        plt.plot(iter[start:end], result_ave[start:end], mk, color=color, label=strgy, markevery=max_evals // 10)
        plt.fill_between(iter[start:end], result_ave[start:end] + error_bar[start:end],
                         result_ave[start:end] - error_bar[start:end], color=color, alpha=0.2)
        plt.xlabel('Number of Evaluations', fontdict={'weight': 'normal', 'size': 8})
        plt.ylabel('Fval(' + str(num_trial) + ')', fontdict={'weight': 'normal', 'size': 8})
        ax = plt.gca()
        ax.yaxis.get_major_formatter().set_powerlimits((-2, 3))
        if truncation <= 700:
            ax.xaxis.set_major_locator(MultipleLocator(100))
        elif truncation == 1000:
            ax.xaxis.set_major_locator(MultipleLocator(200))
        plt.xlim([start, end])
        plt.tick_params(labelsize=8)
        plt.title(current_prob + ' ' + str(prob_dim) + '-D', fontdict={'weight': 'normal', 'size': 8})
        plt.legend(fontsize=8)

        iplot += 1

    dict_fval_best[prob] = fval_best
    qap = [abs(fval - fval_best) / (abs(fval_best) + 1e-12) for fval in xlsx_row]
    xlsx_col_prob.append(current_prob)
    xlsx_col_fval.append(xlsx_row)
    xlsx_col_qap.append(qap)

    # print(ax.get_ylim())
    # ax.set_ylim(fval_best-0.02*(ylim_ub-fval_best), ylim_ub-0.2*(ylim_ub-fval_best))
    ax.set_ylim(fval_best - 0.02 * (ylim_ub - fval_best), ylim_ub)

# Excel writing
workbook = xlsxwriter.Workbook('./result.xlsx')
headrow_format = workbook.add_format({
    # 'bold':     True,
    # 'border':   6,
    'align': 'center',  # 水平居中
    'valign': 'vcenter',  # 垂直居中
    # 'fg_color': '#D7E4BC',#颜色填充
})

float_format = workbook.add_format({
    # 'bold':     True,
    # 'border':   6,
    'align': 'right',  # 右对齐
    'valign': 'vcenter',  # 垂直居中
    'num_format': '0.00',
    # 'fg_color': '#D7E4BC',#颜色填充
})

percentage_format = workbook.add_format({
    'align': 'right',
    'valign': 'vcenter',
    'num_format': '0.00%',
})

worksheet = workbook.add_worksheet()
# Headrow
start = 'B'  # start from B1
for i in range(len(set_strategy)):
    worksheet.merge_range(chr(ord(start) + 2 * i) + '1:' + chr(ord(start) + 2 * i + 1) + '1',
                          set_strategy[i], headrow_format)
    worksheet.write_column('A3:A' + str(3 + nprob - 1),
                           xlsx_col_prob)
    worksheet.write_column(chr(ord(start) + 2 * i) + '3:' + chr(ord(start) + 2 * i) + str(3 + nprob - 1),
                           np.array(xlsx_col_fval)[:, i].tolist(), float_format)
    worksheet.write_column(chr(ord(start) + 2 * i + 1) + '3:' + chr(ord(start) + 2 * i + 1) + str(3 + nprob - 1),
                           np.array(xlsx_col_qap)[:, i].tolist(), percentage_format)
    worksheet.write_formula(chr(ord(start) + 2 * i + 1) + str(3 + nprob),
                            'SUM(' + chr(ord(start) + 2 * i + 1) + '3:' + chr(ord(start) + 2 * i + 1) + str(
                                3 + nprob - 1) + ')',
                            percentage_format)

worksheet.write('A' + str(3 + nprob), 'Q(A)')
worksheet.write_row('B2', ['fval', 'Q(A,P)'] * len(set_strategy), headrow_format)

workbook.close()
#plt.show()

# Accuracy level
tau = 5e-2
nstrgy = len(set_strategy)
tps = np.empty((nprob, nstrgy))
rps = np.empty((nprob, nstrgy))
# print(nstrgy)

rowid = 0
for prob in set_prob:
    # print("Prob: "+prob)
    fval_best = dict_fval_best[prob]
    # print(fval_best)
    tps_best = float('inf')
    colid = 0
    for strgy in set_strategy:
        # print("Strgy: "+strgy)
        for test_result in result:
            # print(test_result.strgy)
            if prob != test_result.prob + str(test_result.prob_dim):
                continue
            if strgy == "TuRBO":
                if "TuRBO" not in test_result.strgy:
                    continue
            else:
                if strgy != test_result.strgy:
                    continue
            current_prob = test_result.prob
            prob_dim = test_result.prob_dim
            strgy = test_result.strgy
            max_evals = min(test_result.max_evals, truncation)
            num_trial = test_result.num_trial
            num_init = test_result.num_init

            fX = test_result.fX
            fX = np.minimum.accumulate(fX, axis=1)
            result_ave = np.mean(fX, axis=0)[0:max_evals]

            fval = np.min(result_ave, axis=0)
            if strgy == 'DYCORS':
                f0 = result_ave[2 * prob_dim + 1]
            elif strgy == 'SDSGCK_SLHD':
                f0 = result_ave[2 * prob_dim + 1]
            elif strgy == 'CK' or strgy == 'SDSG' or strgy == 'SDSGCK' or strgy == 'SDSGDY' or \
                    strgy == 'SDSGCK' or strgy == 'SDSGDYCK' or strgy == 'RBFOpt':
                f0 = result_ave[round(0.5 * (prob_dim + 1)) - 1 if prob_dim < 20 else round(0.4 * (prob_dim + 1)) - 1]
            elif strgy == 'TuRBO_1':
                f0 = result_ave[2 * prob_dim - 1]
            if f0 - fval < (1 - tau) * (f0 - fval_best):
                tps[rowid, colid] = float('inf')
            else:
                tps[rowid, colid] = max_evals
                for i in range(max_evals):
                    if f0 - result_ave[i] >= (1 - tau) * (f0 - fval_best):
                        tps[rowid, colid] = i + 1
                        break

            # print(tps[rowid, colid])

            if tps[rowid, colid] < tps_best:
                tps_best = tps[rowid, colid]

            colid += 1

    # print(prob)
    # print(tps_best)
    rps[rowid, :] = tps[rowid, :] / tps_best
    tps[rowid, :] /= prob_dim + 1
    rowid += 1

# print("rps:")
# print(rps)

# print("tps:")
# print(tps)


# alpha = [i/10 for i in range(10,51,2)]
# alpha = np.sort(np.unique(rps))

rps_max = 5.0

fig = plt.figure(figsize=(9, 3))
# fig = plt.figure()
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.124, right=0.95, hspace=0.2, wspace=0.24)
fig.add_subplot(1, 2, 1)
strgyid = 0
for strgy in set_strategy:
    alpha = np.sort(np.unique(rps[:, strgyid]))
    alpha = np.insert(alpha, 0, 1.0)
    if alpha[-1] == float('inf'):
        alpha = np.delete(alpha, -1)
    if alpha[-1] > rps_max:
        rps_max = alpha[-1] + 0.5
    alpha = np.append(alpha, rps_max)
    rhos = [np.where(rps[:, strgyid] <= a)[0].shape[0] / nprob for a in alpha]

    if strgy == 'DYCORS':
        mk = marker[0]
        color = colors[0]
    elif strgy == 'TuRBO':
        mk = marker[1]
        color = colors[1]
    elif strgy == 'RBFOpt':
        mk = marker[2]
        color = colors[2]
    elif strgy == 'SDSGDYCK':
        mk = marker[3]
        color = colors[3]
        strgy = "DOSS(cnt)"

    elif strgy == 'CK':
        mk = marker[4]
        color = colors[4]
    elif strgy == 'SDSG':
        mk = marker[5]
        color = colors[5]
        strgy = "SG"
    elif strgy == 'SDSGCK':
        mk = marker[6]
        color = colors[6]
        strgy = "EDOSS"
    elif strgy == 'SDSGDY':
        mk = marker[7]
        color = colors[7]
    elif strgy == 'SDSGDYCK_std':
        mk = marker[7]
        color = colors[7]
        strgy = "DOSS(std)"
    elif strgy == 'SDSGCK_SLHD':
        mk = marker[8]
        color = colors[8]
        strgy = "DOSS(SLHD)"
    elif strgy == 'MADS':
        mk = marker[9]
        color = colors[9]
        strgy = "MADS-VNS"
    elif strgy == 'GA':
        mk = marker[4]
        color = colors[4]
        strgy = "GA"
    elif strgy == 'CMA-ES':
        mk = marker[5]
        color = colors[5]
        strgy = "CMA-ES"

    # plt.plot(alpha, rhos, mk, label=strgy)
    #plt.xlim([1.0, rps_max - 0.5])
    plt.xlim([1.0, 5.0])
    plt.ylim([0, 1])
    plt.step(alpha, rhos, mk, color=color, label=strgy, where="post")
    plt.xlabel('Performance Ratio', fontdict={'weight': 'normal', 'size': 12})
    plt.ylabel('Fraction of Instances', fontdict={'weight': 'normal', 'size': 12})

    plt.tick_params(labelsize=8)
    plt.title('Performance Profile on '+str(nprob)+' Problems ' + r"($\tau$=" + str(tau) + ')', fontdict = {'weight': 'normal', 'size': 12})
    plt.legend(fontsize=8)

    # plt.grid()

    strgyid += 1
# plt.grid()
# plt.show()

# alpha = list(range(0, 51, 2))
# alpha = np.sort(np.unique(tps))
# print(alpha)

fig.add_subplot(1, 2, 2)
strgyid = 0
for strgy in set_strategy:
    alpha = np.sort(np.unique(tps[:, strgyid]))
    alpha = np.insert(alpha, 0, 0)
    if alpha[-1] == float('inf'):
        alpha[-1] = 55
    else:
        alpha = np.append(alpha, 55)
    # if alpha[-1] != 50:
    #     np.append(alpha, 50)
    # print("alpha")
    # print(alpha)

    ds = [np.where(tps[:, strgyid] <= a)[0].shape[0] / nprob for a in alpha]
    # print("ds")
    # print(ds)

    if strgy == 'DYCORS':
        mk = marker[0]
        color = colors[0]
    elif strgy == 'TuRBO':
        mk = marker[1]
        color = colors[1]
    elif strgy == 'RBFOpt':
        mk = marker[2]
        color = colors[2]
    elif strgy == 'SDSGDYCK':
        mk = marker[3]
        color = colors[3]
        strgy = "DOSS(cnt)"

    elif strgy == 'CK':
        mk = marker[4]
        color = colors[4]
    elif strgy == 'SDSG':
        mk = marker[5]
        color = colors[5]
        strgy = "SG"
    elif strgy == 'SDSGCK':
        mk = marker[6]
        color = colors[6]
        strgy = "EDOSS"
    elif strgy == 'SDSGDY':
        mk = marker[7]
        color = colors[7]
    elif strgy == 'SDSGDYCK_std':
        mk = marker[7]
        color = colors[7]
        strgy = "DOSS(std)"
    elif strgy == 'SDSGCK_SLHD':
        mk = marker[8]
        color = colors[8]
        strgy = "DOSS(SLHD)"

    elif strgy == 'MADS':
        mk = marker[9]
        color = colors[9]
        strgy = "MADS-VNS"

    elif strgy == 'GA':
        mk = marker[4]
        color = colors[4]
        strgy = "GA"
    elif strgy == 'CMA-ES':
        mk = marker[5]
        color = colors[5]
        strgy = "CMA-ES"



    # plt.plot(alpha, ds, mk, label=strgy)
    plt.xlim([0, 50])
    plt.ylim([0, 1])
    plt.step(alpha, ds, mk, color=color, label=strgy, where="post")
    plt.xlabel('Number of Equivalent Gradient Iterations', fontdict={'weight': 'normal', 'size': 12})
    plt.ylabel('Fraction of Instances', fontdict={'weight': 'normal', 'size': 12})

    plt.tick_params(labelsize=8)
    plt.title('Data Profile on '+str(nprob)+' Problems ' + r"($\tau$="+str(tau)+')', fontdict={'weight': 'normal', 'size': 12})
    plt.legend(fontsize=8)

    strgyid += 1


# plt.grid()
filename = "./figure/Fig1_profile%s.pdf" % str(tau)
plt.savefig(filename)
