function inp = test_getinput(image, meta,buckets)

    audfile 	= [image(1:end-3),'wav'];
    %contructing a vector
	z         	= audioread(audfile); % z = [y, fs]
    
    
    
    
    
    SPEC 		= runSpec( z, meta.audio); % spectrogram
	mu    		= mean(SPEC,2);
    stdev 		= std(SPEC,[],2) ;
    nSPEC 		= bsxfun(@minus, SPEC, mu);
    nSPEC 		= bsxfun(@rdivide, nSPEC, stdev);
    %applies an element-by-element binary operation to arrays a and b, with singleton expansion enabled.
    %basically normalization
    % 512 678
    % size(nSPEC) 512 160
    rsize 	= buckets.width(find(buckets.width(:)<=size(nSPEC,2),1,'last'));
    % find the index of the largest value in buckets smaller than length of nspec
    %rstart  = round((size(nSPEC,2)-rsize)/2);
    rstart = 1;

    inp(:,:) = gpuArray(single(nSPEC(:,rstart:rstart+rsize-1)));
    %size(inp)

end 

