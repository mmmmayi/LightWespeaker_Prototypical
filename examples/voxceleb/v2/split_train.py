import kaldiio
import numpy as np
import matplotlib.pyplot as plt
'''
path = '/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/embeddings/vox1_O/phovector_0s.scp'
#enrol = 'id10299/SX-117N_MoI/00006.wav'
#test = 'id10299/SX-117N_MoI/00004.wav'
enrol = 'id10270/GWXujl-xAVM/00014.wav'
test = 'id10282/YmJBoCojXf4/00001.wav'
#pho = [3,6,23]
pho = [3,6,7]
d = kaldiio.load_scp(path)
enrol = np.transpose(d[enrol])

test = np.transpose(d[test])
cosine_similarity = np.matmul(enrol,test.T)
np.save('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/embeddings/vox1_O/non-target', cosine_similarity)
for i in pho:
    print(cosine_similarity[i][i])
'''
dict_raw = []
with open('pho_list', 'r') as f:
    for line in f:
        phoneme = line.split(' ')[0].strip()
        if phoneme=='[SIL]':
            phoneme='[N-V]'
        dict_raw.append(phoneme)

lb = np.load('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/scores/clean.kaldi1ratio_pho.npy')
se = np.load('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/scores/sitw_eval.kaldi1ratio_pho.npy')
sd = np.load('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/scores/sitw_dev.kaldi1ratio_pho.npy')
vo = np.load('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/scores/vox1_O_cleaned.kaldi1ratio_pho.npy')
ve = np.load('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/scores/vox1_E_cleaned.kaldi1ratio_pho.npy')
vh = np.load('/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/scores/vox1_H_cleaned.kaldi1ratio_pho.npy')

lb=lb[:39]
se=se[:39]
sd=sd[:39]
vo=vo[:39]
ve=ve[:39]
vh=vh[:39]

stacked_arrays = np.stack((lb, se, sd, vo, ve, vh))
mean_array = np.mean(stacked_arrays, axis=0)
#np.save('/home/svu/ma_yi/test/wespeaker/examples/voxceleb/v2/exp/dscrmn',mean_array)

sd_indices = np.argsort(mean_array)[::-1]
lb_sorted = lb[sd_indices]
se_sorted = se[sd_indices]
sd_sorted = sd[sd_indices]
vo_sorted = vo[sd_indices]
ve_sorted = ve[sd_indices]
vh_sorted = vh[sd_indices]
mean_sorted = mean_array[sd_indices]
dimension_names_sorted = np.array(dict_raw)[sd_indices]
x = np.arange(1, 40)
plt.figure(figsize=(6.4, 5.6))
plt.plot(x, lb_sorted, 'o', label='Librispeech', color='blue')
plt.plot(x, se_sorted, 'o', label='SITW-eval', color='red')
plt.plot(x, vo_sorted, 'o', label='Vox1-O', color='green')
plt.plot(x, ve_sorted, 'o', label='Vox1-E', color='purple')
plt.plot(x, vh_sorted, 'o', label='Vox1-H', color='orange')
plt.plot(x, sd_sorted, 'o', label='SITW-dev', color='pink')
plt.plot(x, mean_sorted, 'o', label='Mean', color='black')
plt.xticks(ticks=x, labels=dimension_names_sorted, rotation=90)
plt.xlabel('Phone')
plt.ylabel('F-ratio')
plt.legend()
plt.savefig('test.png')
