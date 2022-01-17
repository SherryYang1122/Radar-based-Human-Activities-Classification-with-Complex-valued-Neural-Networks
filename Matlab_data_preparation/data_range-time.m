% set clear
clc; clear all; close all;
set(0,'DefaultAxesFontSize',14);
set(0,'DefaultFigureWindowStyle','docked');
set(0,'Defaultlinelinewidth',2);
%% range-time
% data address
addr1 = {'wal', 'WandS', 'sit', 'BfSi', 'BfSt', 'WandF', 'FandS'};
addr2 = {'_Fr2','_Har','_Mat', '_Max', '_Nic','_Nik','_Ro2','_Ron','_Sen','_Sim','_Wie','_Xim','_Yic','_Yua','_Yub'};
label = cell(80000,1);
rt_complex = cell(80000,1);
%% calculate spectrogram(STFT) and collect
%parameters
sampletime = 0.0082;    % PRF
range = [1;5.38];       % area of coverage
fs_slow =  1/sampletime; 
%%
% only considering Radar3
count = 0;
rt_len = 480;
for m = 1:28
    j = ceil(m/4);
    %m
    for k = 1:15
        if m < 10
            load(['00',num2str(m),'_mon_',addr1{j}, addr2{k},'.mat']);
        else
            load(['0',num2str(m),'_mon_',addr1{j}, addr2{k},'.mat']);
        end
        range_time = hil_resha_aligned(:,:,3);
        lbl_diff = diff(lbl_out);
        i = 1;
        lal_var = 1;
        while i < size(range_time,2)
            if lbl_diff(i) ~= 0 || (i - lal_var) >= rt_len
                if lbl_out(floor((i+lal_var)/2)) == 0 || (i - lal_var) < 90
                    lal_var = i;
                    i = i + 1;
                else
                    count = count + 1;
                    label{count} = lbl_out(floor((i+lal_var)/2));
                    if ceil((i+lal_var)/2-rt_len/2) < 0
                        rt_complex{count} = imresize(range_time(:,1:rt_len),[240,120],'bilinear');
                    else
                        if ceil((i+lal_var)/2+rt_len/2) > size(range_time,2)
                            rt_complex{count} = imresize(range_time(:,size(range_time,2)-rt_len+1:size(range_time,2)),[240,120],'bilinear');
                        else
                            rt_complex{count} = imresize(range_time(:,ceil((i+lal_var)/2-rt_len/2+1):ceil((i+lal_var)/2+rt_len/2)),[240,120],'bilinear');
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
    rt_complex{index(i)} = reshape(rt_complex{index(i)},1,[]);
end
rt_complex2 = cell2mat(rt_complex(index));
writeNPY([real(rt_complex2),imag(rt_complex2),label_2], 'rt_complex2.npy');