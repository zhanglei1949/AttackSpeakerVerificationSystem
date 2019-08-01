function  dists = demo_vggvox_verif_voxceleb2_batch(arr, varargin)
    % DEMO_VGGVOX_VERIF - minimal demo with the VGGVox model pretrained on the
    % VoxCeleb2 dataset for Speaker Verification
    
    % downloads the VGGVox model and
    % prints the distance on a test evalutation pair
    opts.modelPath = '' ;
    opts.gpu = 1;
    opts.dataDir = 'testfiles/verif'; 
    opts = vl_argparse(opts, varargin) ;
    
    % Example speech segments for input
    %inpPath1 = fullfile(opts.dataDir, '8jEAjG6SegY_0000008.wav');
    %inpPath2 = fullfile(opts.dataDir, 'x6uYqmx31kE_0000001.wav'); 
    
    % Load or download the VGGVox model for Verification pretrained on VoxCeleb2
    modelName = 'ver_net.mat' ;
    paths = {opts.modelPath, ...
        modelName, ...
        fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
    %paths
    ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;
    
    if isempty(ok)
        fprintf('Downloading the VGGVox model for Verification ... this may take a while\n') ;
        opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
        mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
        url = sprintf('%s/~vgg/data/voxceleb2/%s', base, modelName) ;
        urlwrite(url, opts.modelPath) ;
    else
        opts.modelPath = paths{ok} ;
    end
    load(opts.modelPath); 
    net = dagnn.DagNN.loadobj(netStruct);
    
    
    
    %% Audio settings. 
     opt.audio.window   = [0 1];
     opt.audio.fs       = 16000;
     opt.audio.Tw       = 25;
     opt.audio.Ts       = 10;            % analysis frame shift (ms)
     opt.audio.alpha    = 0.97;          % preemphasis coefficient
     opt.audio.R        = [];  % frequency range to consider
     opt.audio.M        = 40;            % number of filterbank channels
     opt.audio.C        = [];            % number of cepstral coefficients
     opt.audio.L        = [];            % cepstral sine lifter parameter%keyboard;
    
    
    net.meta = opt; 
    
    % add a distance layer 
    net.addLayer('dist', dagnn.PDist('p',2), {'x0_s1', 'x0_s2'}, 'distance');
    
    % Evaluate network on GPU and set it to test mode
    net.move('gpu');
    net.conserveMemory = 0;
    net.mode = 'test' ;
    %net.print()
    %net.layers(1)
    %Get network information
    
    % Setup buckets to allow for average pooling 
    
    %net.layers(ind2+1)
    featid = strcmp({net.vars.name},'distance');
    %featid1 = strcmp({net.vars.name}, 'x0_s1');
    %featid2 = strcmp({net.vars.name}, 'x0_s2');
    %net.vars.name
    %size(net.vars.name)
    %buckets.pool 	= [2 5 8 11 14 17 20 23 27 30];
    %buckets.width 	= [100 200 300 400 500 600 700 800 900 1000];
    buckets.pool 	= [2 3 8 11 14 17 20 23 27 30];
    buckets.width 	= [100 160 300 400 500 600 700 800 900 1000];
    dists = {};
    inp1 = test_getinput(arr{1,1}, net.meta, buckets);
    s1 = size(inp1,2);
    p1 = buckets.pool(s1==buckets.width);
    ind1 = net.getLayerIndex('pool_time_b1');
    net.layers(ind1).block.poolSize=[1 p1];
    for c = 2:size(arr, 2)
        % Load input pair and do a forward pass
        inp2 = test_getinput(arr{1,c}, net.meta, buckets);
        %s1 = 600
        s2 = size(inp2,2);
        p2 = buckets.pool(s2==buckets.width);
        % p1 = 17
        %different pooling size for different input size
        %ind1 = 192
        %net.layers(ind1)
        ind2 = net.getLayerIndex('pool_time_b2'); 
        %ind2 = 384
        
        net.layers(ind2).block.poolSize=[1 p2];

        net.eval({ 'data_b1', gpuArray(inp1) ,'data_b2', gpuArray(inp2)});
        dist = gather(squeeze(net.vars(featid).value));
        dists = [dists, dist];
    %score1 = gather((net.vars(featid1).value));
    %score2 = gather((net.vars(featid2).value));
    
    %size(score1)
    %size(score2)
    % Print distance
        %fprintf('dist: %05d \n',dist); % should output a small distance if the two segments come from the same identity
    end
    