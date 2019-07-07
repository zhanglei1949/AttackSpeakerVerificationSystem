function remapped_hearing_thr = calc_threshold_lei(wav_file, win_len, overlap, destination)
% input: wav file name
%       window length
%       overlap between two window, win_len - hop_length
%       destination : where to store the thresholds
addpath('MPEG1')
[audio, fs] = audioread(wav_file);
win_len_ = fs * win_len;
overlap_ = fs * overlap;
[remapped_hearing_thr, remapped_hearing_thr_db] = calculate_hearing_threshold(audio, fs, win_len_, overlap_);

csvwrite(destination, remapped_hearing_thr);

%return remapped_hearing_thr, remapped_hearing_thr_db