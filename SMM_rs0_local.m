
%%%%%%%%%%%%%%%%%%%%%
% Environment setup %
%%%%%%%%%%%%%%%%%%%%%

%-------------%
% Basic setup %
%-------------%

% Risk category used
OtherParams.riskscore = "0";
% Threshold for GPU run time, beyond which unselect this GPU
OtherParams.GPUTimeThreshold = 6;
% Treatment grouops that are simulated
OtherParams.GroupsSimulated = "LowHighMU";

%-----------------------------------------%
% Parallel & GPU management & File pathes %
%-----------------------------------------%

for g=1:gpuDeviceCount()
    device = gpuDevice(g); reset(device);
end
global TotalIter; TotalIter = 0;
global GPUTime; GPUTime = zeros(gpuDeviceCount(),2);
global FreeGPU; FreeGPU = 1:gpuDeviceCount();
global PoolType; 
run patchJobStorageLocation;

if isempty(gcp('nocreate')); pool=parpool(gpuDeviceCount()*4); end;


% Outputs

local_output = 'Estimates/wisdom_rs0.mat'; 
finalestimate = 'Estimates/rs0.mat';

% Inputs

if OtherParams.riskscore=="0"
    EmpiricalMoments = table2array(readtable('SampleMoments/rs0_EmpiricalMoments.xls','ReadVariableNames',false));
    CovMatrix = table2array(readtable('SampleMoments/rs0_VarCovMat.xlsx','ReadVariableNames',false));
elseif OtherParams.riskscore=="0333"
    EmpiricalMoments = table2array(readtable('SampleMoments/rs0333_EmpiricalMoments.xls','ReadVariableNames',false));
    CovMatrix = table2array(readtable('SampleMoments/rs0333_VarCovMat.xlsx','ReadVariableNames',false));
elseif OtherParams.riskscore=="0666"
    EmpiricalMoments = table2array(readtable('SampleMoments/rs0666_EmpiricalMoments.xls','ReadVariableNames',false));
    CovMatrix = table2array(readtable('SampleMoments/rs0666_VarCovMat.xlsx','ReadVariableNames',false));
elseif OtherParams.riskscore=="1"
    EmpiricalMoments = table2array(readtable('SampleMoments/rs1_EmpiricalMoments.xls','ReadVariableNames',false));
    CovMatrix = table2array(readtable('SampleMoments/rs1_VarCovMat.xlsx','ReadVariableNames',false));
end

%--------------------------%
% Non-estimated parameters %
%--------------------------%

% Firm discount rate
OtherParams.disc_rate_firm = 0.9957;
% strength of lock
OtherParams.lambda = 1;
% cash price if pay in full at the beginning
OtherParams.cashprice = 200;
% # of periods for simulations
OtherParams.Periods = 52*2;

%-----------------------------------------%
% Indicators for computation & simulation %
%-----------------------------------------%

% Convergence threshold for VF
OtherParams.convergence_crit1 = 10^(-6);
% parameters used for Tauchen method
OtherParams.GridWidth = 2;
% Whether the dynamics of customers (i.e., post Day 0) are simulated
OtherParams.IF_DynamicSim = 1; 
% Whether firm profit is solved for
OtherParams.IF_SolveProfit = 0;
% Whether welfare measures are calculated
OtherParams.IF_CalcWelfare = 0;
% Whether to inflate the grid (for plotting & comparative statics etc.)
OtherParams.IF_GreaterGridSize = 0;
% Whether doing comparative statics
OtherParams.IF_ComparativeStatics = 0;
% Whether to calculate statistics for the decomposition exercise
OtherParams.IF_DecompMHAS = 0;
% Whether to export simulated datasets
OtherParams.IF_ExportData = 0;
% Whether to reuse the value of being able to buy with cash plus extra cash
OtherParams.IF_LoadOutsideCashAnytimePlusExtraCash = 1;
% Whether to skip loading the outcome option plus cash
OtherParams.IF_SkipLoadingOutsideCashAnytimePlusExtraCash = 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization for SMM %
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Rounds of global/local optimization

N_star = 100;

% Options for Nelder-Mead algorithm

MaxFunEvals = 500 ;
fmoption    = optimset('MaxFunEvals',MaxFunEvals,'Display','iter') ;

% Empirical moments and weighting matrix

EmpiricalMoments = EmpiricalMoments([...
    (34*0+1):(34*0+4) (34*0+5):(34*0+8) (34*0+25):(34*0+27) 34*0+29 34*0+32 ...
    (34*2+1):(34*2+4) (34*2+5):(34*2+8) (34*2+25):(34*2+27) 34*2+29 34*2+32 ...
    (34*4+1):(34*4+4) (34*4+5):(34*4+8) (34*4+25):(34*4+27) 34*4+29 34*4+32 ...
    (34*6+1):(34*6+4) (34*6+5):(34*6+8) (34*6+25):(34*6+27) 34*6+29 34*6+32 ...
    ]);
CovMatrix = CovMatrix([...
    (34*0+1):(34*0+4) (34*0+5):(34*0+8) (34*0+25):(34*0+27) 34*0+29 34*0+32 ...
    (34*2+1):(34*2+4) (34*2+5):(34*2+8) (34*2+25):(34*2+27) 34*2+29 34*2+32 ...
    (34*4+1):(34*4+4) (34*4+5):(34*4+8) (34*4+25):(34*4+27) 34*4+29 34*4+32 ...
    (34*6+1):(34*6+4) (34*6+5):(34*6+8) (34*6+25):(34*6+27) 34*6+29 34*6+32 ...
    ], ...
    [...
    (34*0+1):(34*0+4) (34*0+5):(34*0+8) (34*0+25):(34*0+27) 34*0+29 34*0+32 ...
    (34*2+1):(34*2+4) (34*2+5):(34*2+8) (34*2+25):(34*2+27) 34*2+29 34*2+32 ...
    (34*4+1):(34*4+4) (34*4+5):(34*4+8) (34*4+25):(34*4+27) 34*4+29 34*4+32 ...
    (34*6+1):(34*6+4) (34*6+5):(34*6+8) (34*6+25):(34*6+27) 34*6+29 34*6+32 ...
    ]);
CovMatrix = diag(diag(CovMatrix));
WeightingMatrix = inv(CovMatrix);

% Boundaries

% Lower bound                       % Upperbound        
LowerBound(1)   = 10 ;              UpperBound(1)   = 45 ;       % mean initial weekly income
LowerBound(2)   = 10^-3 ;           UpperBound(2)   = 1.6 ;      % std dev of initial income
LowerBound(3)   = 10^-3 ;           UpperBound(3)   = 0.8;       % sd of income shock
LowerBound(4)   = 0 ;               UpperBound(4)   = 0;         % persistence of income
LowerBound(5)   = 10 ;              UpperBound(5)   = 40;        % mean of initial value, as ratio to average MU
LowerBound(6)   = 0 ;               UpperBound(6)   = 0;         % SD of initial value
LowerBound(7)   = 0.005 ;           UpperBound(7)   = 0.15;      % probability of depreciation
LowerBound(8)   = 1;                UpperBound(8)   = 1;         % risk aversion
LowerBound(9)   = .97 ;             UpperBound(9)   = .998;      % discount rate
LowerBound(10)  = 0 ;               UpperBound(10)  = 8;         % marginal life-term value of liquidity, as ratio to average MU
LowerBound(11)  = 1 ;               UpperBound(11)  = 1;         % convexity of marginal life-term value of liquidity
LowerBound(12)  = 0 ;               UpperBound(12)  = 300;       % random utility shock for 3 month
LowerBound(13)  = 1 ;               UpperBound(13)  = 1;         % random utility shock for 6 month, relative to 3 month
LowerBound(14)  = 1 ;               UpperBound(14)  = 1;         % random utility shock for 9 month, relative to 3 month
LowerBound(15)  = 1 ;               UpperBound(15)  = 1;         % random utility shock for 12 month, relative to 3 month
LowerBound(16)  = -350 ;            UpperBound(16)  = 350;       % fixed effect for 3 month
LowerBound(17)  = -350 ;            UpperBound(17)  = 350;       % fixed effect for 6 month
LowerBound(18)  = -350 ;            UpperBound(18)  = 350;       % fixed effect for 9 month
LowerBound(19)  = 0 ;               UpperBound(19)  = 0;         % fixed effect for 12 month

nParams = size(LowerBound,2) ;

% Intialize output data

wisdom = inf*ones(N_star,nParams+1); % Optimal points from each local stage 



%%%%%%%%%%%%%%%%%%%
% Initial guesses %
%%%%%%%%%%%%%%%%%%%

% Evaluate funciton value at each Sobol points

load('Estimates/guess_all_rs0_global1.mat');
guess_all_part1 = guess_all;
load('Estimates/guess_all_rs0_global2.mat');
guess_all_part2 = guess_all;
load('Estimates/guess_all_rs0_global3.mat');
guess_all_part3 = guess_all;
guess_all = [guess_all_part1;guess_all_part2;guess_all_part3];
guess_all(guess_all(:,1)==-Inf,1) = Inf;

% Load moments and calculate error using the efficient matrix

load('Estimates/guess_all_moments_rs0_global1.mat');
guess_all_moments_part1 = guess_all_moments;
load('Estimates/guess_all_moments_rs0_global2.mat');
guess_all_moments_part2 = guess_all_moments;
load('Estimates/guess_all_moments_rs0_global3.mat');
guess_all_moments_part3 = guess_all_moments;
guess_all_moments = [guess_all_moments_part1;guess_all_moments_part2;guess_all_moments_part3];

for idx=1:length(guess_all_moments)

    SimulatedMoments = nan(4*13,1);
    RawSimulatedMoments = guess_all_moments(idx,2:end);
    SimulatedMoments(13*0+1:13*1) = ...
        [...
        RawSimulatedMoments(0*34+1:0*34+8)'; ...
        RawSimulatedMoments(0*34+25:0*34+27)'; ...
        RawSimulatedMoments(0*34+29); ...
        RawSimulatedMoments(0*34+32)];
    SimulatedMoments(13*1+1:13*2) = ...
        [...
        RawSimulatedMoments(1*34+1:1*34+8)'; ...
        RawSimulatedMoments(1*34+25:1*34+27)'; ...
        RawSimulatedMoments(1*34+29); ...
        RawSimulatedMoments(1*34+32)];
    SimulatedMoments(13*2+1:13*3) = ...
        [...
        RawSimulatedMoments(2*34+1:2*34+8)'; ...
        RawSimulatedMoments(2*34+25:2*34+27)'; ...
        RawSimulatedMoments(2*34+29); ...
        RawSimulatedMoments(2*34+32)];
    SimulatedMoments(13*3+1:13*4) = ...
        [...
        RawSimulatedMoments(3*34+1:3*34+8)'; ...
        RawSimulatedMoments(3*34+25:3*34+27)'; ...
        RawSimulatedMoments(3*34+29); ...
        RawSimulatedMoments(3*34+32)];

    Dev = EmpiricalMoments-SimulatedMoments;
    Error = Dev'*WeightingMatrix*Dev;
    guess_all_moments(idx,1) = Error;

end

% Sort by error

guess_all(:,1) = guess_all_moments(:,1);
guess_all = sortrows(guess_all,1);      
guess_best = guess_all(1:N_star,:);disp(guess_best);




%%%%%%%%%%%%%%%%%%%%%%
% Local optimization %
%%%%%%%%%%%%%%%%%%%%%%

i = 1;

while i<=N_star
            
    % if no local stage has been done
    if i==1
        StructParams = guess_best(i,2:end);
        % if the new starting point is good, perform Nelder-Mead simplex algorithm
        [BestPara, Error] = fminsearchbnd(@(StructParams) ...
            Objective(StructParams,OtherParams,EmpiricalMoments,WeightingMatrix)...
            ,StructParams,LowerBound,UpperBound,fmoption) ;
        wisdom(i,1) = Error ;
        wisdom(i,2:nParams+1) = BestPara ;
        save(local_output,'wisdom');
        i = i+1;
    end
    
    % if local stage has been done
    if i>1
        % get new starting point
        StructParams = guess_best(i,2:end);
        [BestError,BestIdx] = min(wisdom(1:i-1,1));
        weight_p = min(max(0.1,(i/N_star)^0.7),0.995); % weight on the prior best
        StructParams    = weight_p*wisdom(BestIdx,2:end)...
            +(1-weight_p)*StructParams ; % new starting point
        BestPara = wisdom(BestIdx,2:end);
        disp("i is");disp(i);disp("weight on the prior best is");disp(weight_p);
        disp("prior best is");disp(wisdom(BestIdx,2:end));disp("s_i is");disp(guess_best(i,2:end));
        disp("Hence, new starting point is:");disp(StructParams);
        
        % if new starting point yields inf error, go to the next starting point
        % this step diverges from Tik-Tak
        if Objective(StructParams,OtherParams,EmpiricalMoments,WeightingMatrix)==inf
            disp("Not a good starting point. Go to the next one.");
            wisdom(i,1) = inf ;
            wisdom(i,2:nParams+1) = NaN;
            save(local_output,'wisdom');
            i = i+1;
            continue;
        else
            % if the new starting point is good, perform Nelder-Mead simplex algorithm
            disp("Found a good starting point. Start N-M from:"); disp(StructParams);
            [BestPara, Error] = fminsearchbnd(@(StructParams)...
                Objective(StructParams,OtherParams,EmpiricalMoments,WeightingMatrix)...
                ,StructParams,LowerBound,UpperBound,fmoption) ;
            wisdom(i,1) = Error ;
            wisdom(i,2:nParams+1) = BestPara ;
            save(local_output,'wisdom');
            i = i+1;
            continue;
        end

    end
end

% Sort by error

wisdom = sortrows(wisdom,1);            
save(local_output,'wisdom');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A final round of optimization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Threshold for GPU run time, beyond which unselect this GPU
OtherParams.GPUTimeThreshold = 10;
% Whether to inflate the grid (for plotting & comparative statics etc.)
OtherParams.IF_GreaterGridSize = 1;
% Convergence threshold for VF
OtherParams.convergence_crit1 = 10^(-8);
% parameters used for Tauchen method
OtherParams.GridWidth = 3;

delete(gcp('nocreate'))
pool=parpool(gpuDeviceCount()*1);

MaxFunEvals = 2000 ;
fmoption    = optimset('MaxFunEvals',MaxFunEvals,'Display','iter') ;
disp("Starting from"); disp(wisdom(1,2:nParams+1));
disp("A final round of optimization is run, which ends up with:");
[StructParams, Error] = fminsearchbnd(@(StructParams) Objective(StructParams,OtherParams,EmpiricalMoments,WeightingMatrix)...
                ,wisdom(1,2:nParams+1),LowerBound,UpperBound,fmoption) ;
disp(StructParams);disp(Error);

save(finalestimate,'StructParams');