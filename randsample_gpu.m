function y = randsample_gpu(n, k, w)

	%RANDSAMPLE Random sample, with replacement.
	%   Y = RANDSAMPLE(N,K,W) returns a
	%   weighted sample, using positive weights W, taken with replacement.  W is
	%   often a vector of probabilities. 

	sumw = sum(w);
	p = w(:)' / sumw;
	edges = min(cumsum(p),1); % protect against accumulated round-off
	edges(end) = 1; % get the upper edge exact
    edges = repmat(edges,k,1);
    randnum = repmat(rand(k,1,'gpuArray'),1,n);
    [~,y] = max(randnum<edges,[],2);