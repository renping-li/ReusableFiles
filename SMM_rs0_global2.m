
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

global_output = 'Estimates/guess_all_rs0_global2.mat'; 
global_moments_output = 'Estimates/guess_all_moments_rs0_global2.mat'; 

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
OtherParams.convergence_crit1 = 10^(-5);
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

N = 10000;

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

guess_all = nan(N,nParams+1) ; % Initial guesses
guess_all_moments = nan(N,4*34+1) ; % Moments associated with initial guesses



%%%%%%%%%%%%%%%%%%%
% Initial guesses %
%%%%%%%%%%%%%%%%%%%


% Vectors to be tested as initial guesses

gpurng(314);rng(314);
pset   = sobolset(nParams,'Skip',2e6) ;
pset   = scramble(pset,'MatousekAffineOwen') ;
p0  = net(pset,N) ;
SobolParams = LowerBound+(UpperBound-LowerBound).*p0 ;

% Evaluate funciton value at each Sobol points

n = 1;
while n<=N
    StructParams = SobolParams(n,:);
    [Error,RawSimulatedMoments] = Objective(StructParams,OtherParams,EmpiricalMoments,WeightingMatrix);
    guess_all(n,1) = Error ;guess_all(n,2:nParams+1) = StructParams ;
    guess_all_moments(n,1) = Error ;guess_all_moments(n,2:4*34+1) = RawSimulatedMoments ;
    if mod(n,50)==0
        disp("Tried initial shots"); disp(n);
        save(global_output,'guess_all');
        save(global_moments_output,'guess_all_moments');
    end
    n = n+1;
end

save(global_output,'guess_all'); save(global_moments_output,'guess_all_moments');


