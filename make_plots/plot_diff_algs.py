import numpy as np
import matplotlib.pyplot as plt

# *************************  SA *************************
with open("../output-data/compSA&EM-4/reward-R2-sa-1.txt") as f:
    content_sa = f.readlines()

new_list_sa = []
for i in range(len(content_sa)):
    row_sa = content_sa[i]
    row_list = row_sa.split("\t")
    new_list_sa.append(row_list)

list_sa = []
for i in range(len(new_list_sa)):
    new_list = []
    for e in new_list_sa[i]:
        if not e == '\n':
            new_list.append(float(e))
    list_sa.append(new_list)

sa_arr = np.array(list_sa)
mean_sa = np.mean(sa_arr, axis=0)
# print(" mean_sa", mean_sa)

# *************************  EFWD *************************
with open("../output-data/compSA&EM-4/reward-R2-em-.txt") as f:
    content_em = f.readlines()

new_list_em = []
for i in range(len(content_em)):
    row_em = content_em[i]
    row_list = row_em.split("\t")
    new_list_em.append(row_list)

list_em = []
for i in range(len(new_list_em)):
    new_list = []
    for e in new_list_em[i]:
        if not e == '\n':
            new_list.append(float(e))
    list_em.append(new_list)
em_arr = np.array(list_em)
mean_em = np.mean(em_arr, axis=0)
# print(" mean_em", mean_em)

# *************************  EFWD *************************
with open("../output-data/compSA&EM-4/reward-R2-em-phase.txt") as f:
    content_em_ph = f.readlines()

new_list_em_ph = []
for i in range(len(content_em_ph)):
    row_em_ph = content_em_ph[i]
    row_list = row_em_ph.split("\t")
    new_list_em_ph.append(row_list)

list_em_ph = []
for i in range(len(new_list_em_ph)):
    new_list_ph = []
    for e in new_list_em_ph[i]:
        if not e == '\n':
            new_list_ph.append(float(e))
    list_em_ph.append(new_list_ph)
em_ph_arr = np.array(list_em_ph)
mean_em_ph = np.mean(em_ph_arr, axis=0)
# print(" mean_em_ph:  ", mean_em_ph)

# *************************  Random *************************
with open("../output-data/compSA&EM-4/reward-R2-rand-1.txt") as f:
    content_rand = f.readlines()

new_list_rand = []
for i in range(len(content_rand)):
    row_rand = content_rand[i]
    row_list = row_rand.split("\t")
    new_list_rand.append(row_list)

list_rand = []
for i in range(len(new_list_rand)):
    new_list = []
    for e in new_list_rand[i]:
        if not e == '\n':
            new_list.append(float(e))
    list_rand.append(new_list)
rand_arr = np.array(list_rand)
mean_rand = np.mean(rand_arr, axis=0)
# print(" mean_rand", mean_rand)

# *************************  kSP-MDP *************************
with open("../output-data/kSPMdp-k2/rewardkSP-k2.txt") as f:
    content_ksp = f.readlines()

new_list_ksp= []
for i in range(len(content_ksp)):
    row_rand = content_ksp[i]
    row_list = row_rand.split("\t")
    new_list_ksp.append(row_list)

list_ksp = []
for i in range(len(new_list_ksp)):
    new_list = []
    for e in new_list_ksp[i]:
        if not e == '\n':
            new_list.append(float(e))
    list_ksp.append(new_list)
ksp_arr = np.array(list_ksp)
mean_ksp = np.mean(ksp_arr, axis=0)
print(" mean_ksp", mean_ksp)

# ************** Statistics  ******************
convergence_point = 20  # just by looking at the plots

# ************* SA
means_sa_after_converge = mean_sa[convergence_point:]
sa_total_mean_after_converge = np.mean(means_sa_after_converge) / 4  # mean_sa[50:-1]  (np.mean(mean_sa) / 4) * 100
sa_std = np.std(means_sa_after_converge)
print("mean_after_converge_sa:  ", sa_total_mean_after_converge)
print("std_sa:  ", sa_std)

# ************* EFWD
means_em_after_converge = mean_em[convergence_point:]
# print(" means_em_after_converge: ", means_em_after_converge)
em_total_mean_after_converge = np.mean(means_em_after_converge) / 4   # mean_sa[50:-1]   (np.mean(mean_em) / 4) * 100
em_std = np.std(means_em_after_converge)
print(" \n em_total_mean_after_converge:  ", em_total_mean_after_converge)
print(" std_em:  ", em_std)

# ************* EFWD_phase
means_em_ph_after_converge = mean_em_ph[convergence_point:]
# print(" means_em_after_converge: ", means_em_after_converge)
em_ph_total_mean_after_converge = np.mean(means_em_ph_after_converge) / 4   # mean_sa[50:-1]   (np.mean(mean_em) / 4) * 100
em_ph_std = np.std(means_em_ph_after_converge)
print(" \n em_ph_total_mean_after_converge:  ", em_ph_total_mean_after_converge)
print(" em_ph_std:  ", em_ph_std)

# ************* Random
means_rand_after_converge = mean_rand[convergence_point:]
rnd_total_mean_after_converge = np.mean(means_rand_after_converge) / 4   # mean_sa[50:-1]   (np.mean(mean_em) / 4) * 100
rnd_std = np.std(means_rand_after_converge)
print(" \n rnd_total_mean_after_converge:  ", rnd_total_mean_after_converge)
print(" rnd_std:  ", rnd_std)

# ************* Random
means_ksp_after_converge = mean_ksp[convergence_point:]
ksp_total_mean_after_converge = np.mean(means_ksp_after_converge) / 4   # mean_sa[50:-1]   (np.mean(mean_em) / 4) * 100
ksp_std = np.std(means_ksp_after_converge)
print(" \n ksp_total_mean_after_converge:  ", ksp_total_mean_after_converge)
print(" ksp_std:  ", ksp_std)

# *************** Plotting results
plt.plot(mean_em, 'g',  linestyle='-', label='EFWD')  # , 'ro'   , marker='*'
plt.plot(mean_em_ph, 'm', linestyle='--', label='phase-EFWD')  # , 'ro' , marker='+'
plt.plot(mean_sa, linestyle='-.', label='SA')  # , 'ro' , marker='o'
plt.plot(mean_rand, 'r', linestyle='--', marker='+', label='Random')  # , 'ro' ,  marker='>'
plt.plot(mean_ksp, 'k', linestyle=':', label='kSP-MDP')  # , 'ro'  , marker='d'
plt.title("A comparison of self-absorbed and EFWD algorithms on 2*2 grid world")
plt.legend(loc='lower right')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.savefig('../plots/emVsSaVSRnd7.png', bbox_inches='tight')

