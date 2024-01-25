import numpy as np
import scipy
from scipy.signal import filtfilt
from scipy import io


def pre_process(channels, sample_length, sample_interval, subban_no, totalsubject, totalblock, totalcharacter, sampling_rate, dataset):
    # Initialization
    total_channels = len(channels)
    AllData = np.zeros((total_channels, sample_length, subban_no, totalcharacter, totalblock, totalsubject))
    y_AllData = np.zeros((1, totalcharacter, totalblock, totalsubject))

    # Forming bandpass filters
    high_cutoff = np.ones(subban_no) * 90
    # low_cutoff = np.arange(8, 8 * (subban_no + 1), 8)
    low_cutoff = np.arange(9.25, 9.25 * (subban_no + 1), 9.25)
    filter_order = 2
    PassBandRipple_val = 1
    bpFilters = [None] * subban_no

    for i in range(subban_no):
        Wn = [low_cutoff[i] / (0.5 * sampling_rate), high_cutoff[i] / (0.5 * sampling_rate)]
        b, a = scipy.signal.iirfilter(N=filter_order, Wn=Wn, rp=PassBandRipple_val, btype='band', ftype='cheby1')
        #bpFilt1 = butter(filter_order, [low_cutoff[i], high_cutoff[i]], btype='band', fs=sampling_rate)
        bpFilters[i] = (b, a)

    # mat_dir_path='/home/seunghan/miniconda3/envs/test_tc/benchmark/'
    mat_dir_path='/home/heejae/data/wearable/'

    mat_file = io.loadmat('/home/heejae/data/Subjects_Information.mat')
    male_subject_indices = [index for index, subject in enumerate(mat_file['Subjects_Information']) if subject[1][0] == 'Male']
    female_subject_indices = [index for index, subject in enumerate(mat_file['Subjects_Information']) if subject[1][0] == 'Female']
    male_index = np.random.choice(male_subject_indices, 25)
    female_index = np.random.choice(female_subject_indices, 25)
    subject_index = np.concatenate((male_index, female_index), axis=0)



    # Filtering
    subject_indexing = 0
    for subject in subject_index:
        #데이터 로드
        if subject<10:
            nameofdata = f'S00{subject}.mat'
        elif subject<100:
            nameofdata = f'S0{subject}.mat'
        else:
            nameofdata = f'S{subject}.mat'
            
        data = scipy.io.loadmat(mat_dir_path+ nameofdata)
        data = data['data'] #Electrode index’, ‘Time points’, ‘Target index’, and ‘Block index'
        if dataset == 'BETA':
            data = data['EEG']

        data2=np.transpose(np.array(data), axes=(0, 1, 2, 4, 3))
        data2=data2[:,:,1,:,:]
        
        # 각 차원에 대한 인덱스를 준비합니다.
        one = np.array([1])
        channel_indices = np.ix_(channels-one, sample_interval, range(data2.shape[2]), range(6))
        
        # 준비된 인덱스로 데이터를 추출합니다.
        sub_data = data2[channel_indices]
        
        for  chr in range(totalcharacter):
            for blk in range(totalblock):
                if dataset == 'Bench':
                    tmp_raw = sub_data[:, :, chr, blk]
                elif dataset == 'BETA':
                    tmp_raw = sub_data[:, :, blk, chr]

                for i in range(subban_no):
                    processed_signal = np.zeros((total_channels, sample_length))
                    for j in range(total_channels):
                        b, a = bpFilters[i]
                        processed_signal[j, :] = filtfilt(b, a, tmp_raw[j, :])

                    AllData[:, :, i, chr, blk , subject_indexing] = processed_signal
                    y_AllData[0, chr, blk, subject_indexing] = chr

        subject_indexing += 1
    

    return AllData, y_AllData
