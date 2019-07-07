function [LTMin, Delta] = PsychoAcousticModel(Input, NumberOfBands)
    % Main function - sampling rate fs = 44100; bitrate = 128;
    %   Author: 
    %          Fabien A.P. Petitcolas (fapp2@cl.cam.ac.uk)
    %          Computer Laboratory
    %          University of Cambridge
    %   Corrections and improvements:
    %          Teddy Furon (furont@thmulti.com), 
    %          Laboratoire TSI - Telecom Paris
    %          UIIS Lab - Thomson multimedia R&D France 
    %          Michael Arnold (arnold@igd.fhg.de)
    %          Fraunhofer Institute for Computer Graphics (IGD)     
    %   References: 
    %    [1] Information technology -- Coding of moving pictures and associated 
    %      audio for digital storage media at up to 1,5 Mbits/s -- Part3: audio. 
    %      British standard. BSI, London. October 1993. Implementation of 
    %      ISO/IEC 11172-3:1993. BSI, London. First edition 1993-08-01. 
    %   Legal notice: 
    %    This computer program is based on ISO/IEC 11172-3:1993, Information 
    %    technology -- Coding of moving pictures and associated audio for digital 
    %    storage media at up to about 1,5 Mbit/s -- Part 3: Audio, with the 
    %    permission of ISO. Copies of this standards can be purchased from the 
    %    British Standards Institution, 389 Chiswick High Road, GB-London W4 4AL,  
    %    Telephone:+ 44 181 996 90 00, Telefax:+ 44 181 996 74 00 or from ISO, 
    %    postal box 56, CH-1211 Geneva 20, Telephone +41 22 749 0111, Telefax 
    %    +4122 734 1079. Copyright remains with ISO. 
    %---------------------------------------------------------------------------- 
    %
    % [LTmin, Delta] = PsychoAcousticModel(Input, NumberOfBands) computes the
    % minimum masking threshold LTmin from Input vector. NumberOfBands specifies
    % the required frequency resolution.
    %
    % -- INPUT --
    % Input: Row vector of Blocksize (= FFT_SIZE = 512) samples with float values
    % scaled within the range [-1, 1].
    %   
    % NumberOfBands: Integer value. For Blocksize samples this value is
    % of the elements [16 | 32 | 64 | 128 | 256].
    %  
    % -- OUTPUT --
    % LTmin: Column vector with FFT_SIZE/2 elements containing the minium loudness
    % threshold values in dB.
    %  
    % Delta: Delta = 96dB - max(X). Delta is a scalar containing the difference to
    % 96 dB for the input.  
    % ------------
       
    % Define global constants 
    % (loaded from Common_Const.mat and Tables_fs_44100.mat in calling function) 
    % FFT_SIZE = 512: Length of analysis window (Input vector). 
    % MIN_POWER = -200: Used for initialisation to avoid taking log(0).
    %
    % INDEX = 1, BARK = 2, ATH = 3: Column indexes for TH, Tonal_list and
    % Non_tonal_list.
    % SPL = 2: Column indexes for the Tonal_list and Non_tonal_list for Sound
    % Pressure Level. 
     
    % TH is a 106x3 matrix. 
    % TH(:, INDEX): Frequency indexes at the top end of each critical band
    % (corresponding to absolute frequency values of table D.1b (pp. 117), fs =
    % 44.1 kHz).
    % TH(:, BARK): Top end of each critical band rate.
    % TH(:, ATH): Absolute ThresHold in quiet (includes offset of -12dB for bit
    % rates >= 96 kbits/s from table D.1b (pp. 117) for fs = 44.1 kHz)
     
    % NOT_EXAMINED = 0, TONAL = 1, NON_TONAL = 2, IRRELEVANT = 3: Flags
    % describing the component type.
     
    % Map is a row vector with 256 elements. It maps the 106 non-linear frequency
    % coefficients onto the 256 frequency indexes.
     
    % CB is a column vector with 25 elements.
    % CB: It contains the indexes for the top end of each critical band (24 bands)
    % in terms of the 106 indexes (column two of D.2b (pp. 123) for 44.1 kHz).
       
    % LTq: Column vector with 106 elements, approximating absolute threshold. 
    % -------------------------------------------------------------------------
       
    global FFT_SIZE MIN_POWER NOT_EXAMINED IRRELEVANT TONAL NON_TONAL 
    global TH INDEX BARK ATH SPL Map CB LTq 
     
    % Psychoacoustic analysis 
     
    % Compute the FFT for power spectrum estimation [1, pp. 110]. 
    [X, Delta] = FFT_Analysis(Input); 
     
    % Find the tonal (sine like) and non-tonal (noise like) components of the
    % signal [1, pp. 111--113]
    [Flags Tonal_list Non_tonal_list] = Find_tonal_components(X); 
     
    % Decimate the maskers: eliminate all irrelevant maskers [1, pp. 114] 
    [Flags Tonal_list Non_tonal_list] = 
                       Decimation(Tonal_list, ... Non_tonal_list, Flags);
    % Compute the individual masking thresholds [1, pp. 113--114]  
    [LTt, LTn] = Individual_masking_thresholds(X', Tonal_list, Non_tonal_list);  
     
    % Compute the global masking threshold [1, pp. 114] 
    LTg = Global_masking_threshold(LTt, LTn);
     
    if NumberOfBands < FFT_SIZE/2,
      % Determine the minimum masking threshold in each subband of NumberOfBands
      % [1, pp. 114]. 
      LTMin = LTmin(LTg, NumberOfBands);
    else 
      % Map threshold LTg from non-linear to linear frequency indexes.
      LTMin = LTg(Map);
    end
     
    % Transpose row vectors for output
    LTMin = LTMin';
    Delta = Delta';