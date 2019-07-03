function remapped_heating_thr = test_threshold(wav_file)


[audio, fs] = audioread(wav_file);
%[remapped_heating_thr, remapped_heating_thr_db] = calculate_hearing_threshold(audio, fs, 512, 256);
[remapped_heating_thr, remapped_heating_thr_db] = calculate_hearing_threshold(audio, fs, 400, 240);
