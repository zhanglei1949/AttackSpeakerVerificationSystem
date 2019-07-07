function remapped_heating_thr = test_threshold(wav_file)


[audio, fs] = audioread(wav_file);
[remapped_heating_thr,remapped_heating_thr_dB]  = calculate_hearing_threshold(audio, fs, 512, 352);
%remapped_heating_thr = calculate_hearing_threshold(audio, fs, 400, 240);

