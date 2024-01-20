function X = simulate_gpu(mc,numSteps,numSims,w)

% 1st input: dtmc object
% 2nd input: number of periods
% 3rd input: number of buyers
% 4th input: probability (transition?) matrix

    % SIMULATE Simulate Markov chain state walks
    % mc is markov chain
    % w is initial state distribution

    P = mc.P;
    numStates = mc.NumStates;

    X = zeros(numSims,1+numSteps,'gpuArray');
    % simulate uniformly distributed random variables
    U = rand(numSims,numSteps,'gpuArray');
    % initial state
    X(:,1) = randsample_gpu(numStates,numSims,w);
    % new state
    mcP = gpuArray(mc.P);
    [~,v] = max(U(:,1)<cumsum(mcP(X(:,1),:),2),[],2);
    X(:,2) = v;

    % repeat for all periods
    for i = 3:(1+numSteps)
        [~,v] = max(U(:,i-1)<cumsum(mcP(v,:),2),[],2);
        X(:,i) = v;
    end

    X=X';

end