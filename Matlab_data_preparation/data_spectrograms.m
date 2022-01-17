%% set clear
clc; clear all; close all;
set(0,'DefaultAxesFontSize',14);
set(0,'DefaultFigureWindowStyle','docked');
set(0,'Defaultlinelinewidth',2);
%% spectrograms
% data address
addr1 = {'wal', 'WandS', 'sit', 'BfSi', 'BfSt', 'WandF', 'FandS'};
addr2 = {'_Fr2','_Har','_Mat', '_Max', '_Nic','_Nik','_Ro2','_Ron','_Sen','_Sim','_Wie','_Xim','_Yic','_Yua','_Yub'};
label = cell(25000,1);
spec_complex = cell(25000,1);
%% calculate spectrogram(STFT) and collect
%only considering Radar3
%parameters
sampletime = 0.0082;    % PRF
range = [1;5.38];       % area of coverage
frequencybin = 110;
win_size = 244;
win_overl = 240;
nfft = 240; 
fs_slow =  1/sampletime; 
count = 0;
seg_len = 480;
spec_len = floor(seg_len/4);          
for m = 1:28
    j = ceil(m/4);
    %m
    for k = 1:15
        if m < 10
            load(['00',num2str(m),'_mon_',addr1{j}, addr2{k},'.mat']);
        else
            load(['0',num2str(m),'_mon_',addr1{j}, addr2{k},'.mat']);
        end
        specs = fft(hil_resha_aligned(:,:,3));
        frequencybin = 110;
        specs111 = specs(frequencybin,:); % the strongest frequnecy bin is approx. 
        [S_M2,f,t,P_M2] = spectrogram(detrend(specs111,2),...
            hann(win_size),...
            win_overl, ...
            nfft,...
            fs_slow,...
            'yaxis','centered',...
            'MinThreshold',40); % can be adjusted or removed; 
        lbl_diff = diff(lbl_out);
        i = win_overl/2;
        lal_var = 1;
        while i < size(specs111,2)
            if lbl_diff(i) ~= 0 || (i - lal_var) >= seg_len
                if lbl_out(floor((i+lal_var)/2)) == 0 || (i - lal_var) < 90
                    lal_var = i;
                    i = i + 1;
                else
                    count = count + 1;
                    label{count} = lbl_out(floor((i+lal_var)/2));
                    if ceil((i+lal_var)/8-win_overl/8-spec_len/2) < 0
                        spec_complex{count} = S_M2(:,1:spec_len);
                    else
                        if ceil((i+lal_var)/8-win_overl/8+spec_len/2) > size(P_M2,2)
                            spec_complex{count} = S_M2(:,size(P_M2,2)-spec_len+1:size(P_M2,2));
                        else
                            spec_complex{count} = S_M2(:,ceil((i+lal_var)/8-win_overl/8-spec_len/2+1):ceil((i+lal_var)/8-win_overl/8+spec_len/2));
                        end
                    end
                    lal_var = i;
                    i = i + 1;
                end
            else
                i = i + 1;
            end
        end
    end
end
%% some segments are deleted (dataset balanced)
label_mat = cell2mat(label(1:count));
index = [];
for i = 1:9
    e = find(label_mat == i); 
    if length(e) > 1000
        index = [index,randsample(e,1000)'];
    else
        index = [index,e'];
    end
end
%% save as npy (complex)
%After downloading npy-matlab as a zip file or via git, just add the npy-matlab directory to your search path
addpath('/Applications/MATLAB_R2021a.app/npy-matlab')
savepath
label_2 = label_mat(index);
for i = 1:size(index,2)
    spec_complex{index(i)} = reshape(spec_complex{index(i)},1,[]);
end
spec_complex2 = cell2mat(spec_complex(index));
writeNPY([real(spec_complex2),imag(spec_complex2),label_2], 'spec_complex2.npy');