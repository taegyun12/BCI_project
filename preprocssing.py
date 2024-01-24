import numpy as np
import scipy
from scipy.signal import filtfilt
from scipy.io import loadmat, savemat


def pre_process(channels, sample_length, sample_interval, subban_no, totalsubject, totalblock, totalcharacter, sampling_rate, dataset):

    mat_dir_path = '/home/heejae/data/benchmark/'

    # Initialization
    total_channels = len(channels)
    AllData = np.zeros((total_channels, sample_length, subban_no, totalcharacter, totalblock, totalsubject))
    y_AllData = np.zeros((1, totalcharacter, totalblock, totalsubject))

    # Forming bandpass filters
    high_cutoff = np.ones(subban_no) * 90
    low_cutoff = np.arange(8, 8 * (subban_no + 1), 8)
    filter_order = 2
    PassBandRipple_val = 1
    bpFilters = [None] * subban_no

    for i in range(subban_no):
        Wn = [low_cutoff[i] / (0.5 * sampling_rate), high_cutoff[i] / (0.5 * sampling_rate)]
        b, a = scipy.signal.iirfilter(N=filter_order, Wn=Wn, rp=PassBandRipple_val, btype='band', ftype='cheby1')
        bpFilters[i] = (b, a)

    # Filtering
    for subject in range(1, totalsubject + 1):
        #데이터 로드
        nameofdata = f'S{subject}.mat'
        data = scipy.io.loadmat(mat_dir_path+ nameofdata)
        data = data['data'] #Electrode index’, ‘Time points’, ‘Target index’, and ‘Block index'
        if dataset == 'BETA':
            data = data['EEG']

        # 각 차원에 대한 인덱스를 준비합니다.
        one = np.array([1])
        #targets : 9.4 /10.2 /13.6 /15.6 Hz
        targets = [10, 17, 29, 31]
        channel_indices = np.ix_(channels-one, sample_interval, targets, range(data.shape[3]))

        # 준비된 인덱스로 데이터를 추출합니다.
        sub_data = data[channel_indices]

        for chr in range(totalcharacter):
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

                    AllData[:, :, i, chr, blk , subject - 1] = processed_signal
                    y_AllData[0, chr, blk, subject - 1] = chr

    return AllData, y_AllData
