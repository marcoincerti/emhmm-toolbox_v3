import numpy as np
from emhmm_v2.util.read_xls_fixations import read_xls_fixations
from emhmm_v2.hmm.vbhmm_learn import vbhmm_learn

np.random.seed(101)

# Load data from xls
data, SubjNames, TrialNames = read_xls_fixations('emhmm_toolbox_v3/demo/demodata.xls')

K = [2, 3]
vbopt = {
    'alpha': 0.1,
    'mu': np.array([256, 192]),
    'W': 0.005,
    'beta': 1,
    'v': 5,
    'epsilon': 0.1,
    'showplot': 0,
}
faceimg = 'face.jpg'

# Learn Subject's HMMs
N = len(data)
hmms = []
for i in range(N):
    print(f'=== running Subject {i+1} ===')
    hmms.append(vbhmm_learn(data[i], K, vbopt))

# Show subject 1
vbhmm_plot(hmms[0], data[0], faceimg)
plt.figure()
vbhmm_plot_compact(hmms[0], faceimg)

# Plot each subject
plt.figure(figsize=(12, 16))
for i in range(N):
    plt.subplot(4, 3, i+1)
    vbhmm_plot_compact(hmms[i], faceimg)
    plt.title(f'Subject {i+1}')

# Run HEM clustering (1 group)
print('=== Clustering (1 group) ===')
hemopt = {'sortclusters': 'f'}
all_hmms1 = vhem_cluster(hmms, 1, 3, hemopt)
vhem_plot(all_hmms1, faceimg)

# Run HEM Clustering (2 groups)
print('=== Clustering (2 groups) ===')
group_hmms2 = vhem_cluster(hmms, 2, 3, hemopt)
vhem_plot(group_hmms2, faceimg)
vhem_plot_clusters(group_hmms2, hmms, faceimg)

# Show group membership
print('Group membership:')
for j, group in enumerate(group_hmms2['groups']):
    print(f'  group {j+1} = {group}')

# Statistical test
data1 = [data[i] for i in group_hmms2['groups'][0]]
data2 = [data[i] for i in group_hmms2['groups'][1]]

p, info, lld = stats_ttest(group_hmms2['hmms'][0], group_hmms2['hmms'][1], data1)
print(f'test group hmm1 different from group hmm2: t({info["df"]})={info["tstat"]}; p={p}')

p, info, lld = stats_ttest(group_hmms2['hmms'][1], group_hmms2['hmms'][0], data2)
print(f'test group hmm2 different from group hmm1: t({info["df"]})={info["tstat"]}; p={p}')
