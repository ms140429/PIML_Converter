% ==================================================================================================
% Author: Shuai Zhao @ Aalborg University, szh@energy.aau.dk
% Note:
%      * The code and data accompany the paper:
%        S. Zhao, Y. Peng, Y. Zhang, and H. Wang, "Parameter Estimation of Power Electronic
%        Converters with Physics-informed Machine Learning", IEEE Trans. Power Electronics, 2022
%      * This is the code for simulation data generation. Please first run
%        the "BuckConverter.m" for voltage & current generation, and then "dataGeneration.m" for
%        preparating the format of following python code "PIML_Converter.py".
% ==================================================================================================
%% Simulated data
clc; close all; clear all;
rng(10);
load simulationDataset.mat

%% Experiment setting
% add noise
%                    noise level--enableADC--syncError
experimentalSetting=[    0            0         0;... % clear data
                         0            1         0;... % ADC error
                         0            0         10;... % Sync error
                         5            0         0;... % 5 noise
                         10           0         0;... % 10 noise
                         5            1         10;... % ADC-Sync-5noise
                         10           1         10];   % ADC-Sync-10noise
experimentalSettingIdx=7; % will be set as 1-7 for emulating various affecting factors

noiseLevel=experimentalSetting(experimentalSettingIdx,1);

% Manually add noise in the simulated data
IL=max(IL+noiseLevel*10/(2^12-1)*randn(length(IL),1),0);% std(IL)=0.495
Vo=max(Vo+noiseLevel*30/(2^12-1)*randn(length(Vo),1),0); % std(Vo)=1.263

% ADC quantized
ADC_enable=experimentalSetting(experimentalSettingIdx,2);
if ADC_enable==1
    IL=adcQuantize(IL, 12, 0, 10);
    Vo=adcQuantize(Vo, 12, 0, 30);
end

% sync error
syncErrorLevel=experimentalSetting(experimentalSettingIdx,3);% 0(0us) or 10(2us);

%% Data preparation

idxD_nonzero=[0;find(diff(Dswitch)~=0);length(Dswitch)];
idxD_nonzero_RloadChange=[find(diff(Rload)~=0)];
tConst=tout;
DswitchTransform=zeros(length(tout),1);
idxOn=1;idxOff=1;
currentInput=zeros(length(tout),1);voltageInput=zeros(length(tout),1);

timeBias=0.00;
for i=2:1:length(idxD_nonzero)-1
    tout(idxD_nonzero(i)+1:idxD_nonzero(i+1))=max(tout(idxD_nonzero(i)+1:idxD_nonzero(i+1))...
        -tConst(idxD_nonzero(i))+timeBias*1/2e4*randn(length(tout(idxD_nonzero(i)+1:idxD_nonzero(i+1))),1),0);
    DswitchTransform(idxD_nonzero(i)+1:idxD_nonzero(i+1),1)=idxOn;
    currentInput(idxD_nonzero(i)+1:idxD_nonzero(i+1),1)=IL(idxD_nonzero(i));
    voltageInput(idxD_nonzero(i)+1:idxD_nonzero(i+1),1)=Vo(idxD_nonzero(i));
    idxOn=idxOn+1;
end

% Data sampling
fig=figure; set(0,'defaulttextinterpreter','latex');
% sgtitle('Condition monitoring signals');
set(fig,'units','normalized','outerposition',[0 0 1 1]);
subplot(4,1,1); plot(tConst,IL,'b'); ylabel('Inductor current','Fontsize',15);
subplot(4,1,2); plot(tConst,Vo,'b'); ylabel('Output voltage','Fontsize',15);
subplot(4,1,3); plot(tConst,Dswitch,'b'); ylabel('Switch state','Fontsize',15); ylim([0,1.5]);
subplot(4,1,4); plot(tConst,Rload,'b'); ylabel('Load resistance','Fontsize',15); xlabel('Simulation time');

%% Data selection --part I: startup
startPercentage=8.45;%1.76;%3.62;%0.02;%5.5;%0.01;%0.02;%  %0-12;
endPercentage=7; %0-12;
sampleNum=120; %total data point is sampleNum*5;
startPoint=floor(length(idxD_nonzero)/12*startPercentage);
endPoint=floor(length(idxD_nonzero)/12*endPercentage)-2;
samplingIdx=startPoint+1:2:startPoint+sampleNum*2;
idx_ui=[];idx_uv=[];idx_fi=[];idx_fv=[];
forwaredBackwaredIndicator=[];
for j=1:1:sampleNum

    i=samplingIdx(j);
    res=200;%57;%207;%floor((idxD_nonzero(i+2)-idxD_nonzero(i))/3);
    idx_ui=[idx_ui ...
        idxD_nonzero(i) ...
        idxD_nonzero(i+1) ...
        idxD_nonzero(i+1) ...
        idxD_nonzero(i+2)];

    % add forward or backward indicator
    forwaredBackwaredIndicator=[forwaredBackwaredIndicator' [-2 0; ...
        2 0; ...
        -2 1; ...
        2 1]']';
end

%% Data selection --part II: workload change
startPercentage=3.67;%1.76;%3.62;%0.02;%5.5;%0.01;%0.02;%  %0-12;
endPercentage=7; %0-12;
sampleNum=120; %total data point is sampleNum*5;
startPoint=floor(length(idxD_nonzero)/12*startPercentage);
endPoint=floor(length(idxD_nonzero)/12*endPercentage)-2;
samplingIdx=startPoint+1:2:startPoint+sampleNum*2;

for j=1:1:sampleNum

    i=samplingIdx(j);

    res=200;%57;%207;%floor((idxD_nonzero(i+2)-idxD_nonzero(i))/3);

    idx_ui=[idx_ui ...
        idxD_nonzero(i) ...
        idxD_nonzero(i+1) ...
        idxD_nonzero(i+1) ...
        idxD_nonzero(i+2)];

    forwaredBackwaredIndicator=[forwaredBackwaredIndicator' [-2 0; ...
        2 0; ...
        -2 1; ...
        2 1]']';
end

%% Data selection --part III: workload change
startPercentage=6.06;%1.76;%3.62;%0.02;%5.5;%0.01;%0.02;%  %0-12;
endPercentage=7; %0-12;
sampleNum=120; %total data point is sampleNum*5;
startPoint=floor(length(idxD_nonzero)/12*startPercentage);
endPoint=floor(length(idxD_nonzero)/12*endPercentage)-2;
% samplingIdx=sort(startPoint+randperm(floor((endPoint-startPoint)/2),sampleNum)*2);
% samplingIdx=startPoint:floor((endPoint-startPoint)/sampleNum):endPoint;
samplingIdx=startPoint+1:2:startPoint+sampleNum*2;
% idx_ui=[];idx_uv=[];idx_fi=[];idx_fv=[];
% forwaredBackwaredIndicator=[];
for j=1:1:sampleNum

    i=samplingIdx(j);

    res=200;%57;%207;%floor((idxD_nonzero(i+2)-idxD_nonzero(i))/3);

    idx_ui=[idx_ui ...
        idxD_nonzero(i) ...
        idxD_nonzero(i+1) ...
        idxD_nonzero(i+1) ...
        idxD_nonzero(i+2)];

    forwaredBackwaredIndicator=[forwaredBackwaredIndicator' [-2 0; ...
        2 0; ...
        -2 1; ...
        2 1]']';
end

idx_uv=idx_ui;
idx_fi=idx_ui;
idx_fv=idx_ui;
%}

%% 1-Input signals for x_ui
ModelInfo.tInitial=tConst;
ModelInfo.x_ui_idx=idx_ui; idx=ModelInfo.x_ui_idx;
ModelInfo.IL=IL(idx);ModelInfo.Vf=Vf(idx);
ModelInfo.Dswitch=Dswitch(idx);ModelInfo.DswitchTransform=DswitchTransform(idx);
ModelInfo.t=tout(idx);ModelInfo.tConst=tConst(idx);
ModelInfo.Vo=Vo(idx);ModelInfo.Rload=Rload(idx);
ModelInfo.currentInput=currentInput(idx);
ModelInfo.voltageInput=voltageInput(idx);

ModelInfo.x_ui=[ModelInfo.t ModelInfo.currentInput ModelInfo.Dswitch ModelInfo.Rload 1-ModelInfo.Dswitch ModelInfo.voltageInput];
ModelInfo.y_ui=[ModelInfo.IL];%+0.26*(1e-2*randn((length(ModelInfo.Vo)),1));

subplot(4,1,1);hold on;plot(ModelInfo.tConst,ModelInfo.IL,'ro','MarkerSize',3);

%% 2-Input signals for x_uv
ModelInfo.x_uv_idx=idx_uv;idx=ModelInfo.x_uv_idx;
ModelInfo.IL=IL(idx);ModelInfo.Vf=Vf(idx);
ModelInfo.Dswitch=Dswitch(idx);ModelInfo.DswitchTransform=DswitchTransform(idx);
ModelInfo.t=tout(idx);ModelInfo.tConst=tConst(idx);
ModelInfo.Vo=Vo(idx);ModelInfo.Rload=Rload(idx);
ModelInfo.currentInput=currentInput(idx);
ModelInfo.voltageInput=voltageInput(idx);

ModelInfo.x_uv=[ModelInfo.t ModelInfo.currentInput ModelInfo.Dswitch ModelInfo.Rload 1-ModelInfo.Dswitch ModelInfo.voltageInput];
subplot(4,1,2);hold on;plot(ModelInfo.tConst,ModelInfo.Vo,'ro','MarkerSize',3);

ModelInfo.y_uv=[ModelInfo.Vo];

%% 3-Input signals for x_fi
% idx=sort(randomSeq(2*dataNum+1:3*dataNum));
ModelInfo.x_fi_idx=idx_fi;idx=ModelInfo.x_fi_idx;
ModelInfo.IL=IL(idx);ModelInfo.Vf=Vf(idx);
ModelInfo.Dswitch=Dswitch(idx);ModelInfo.DswitchTransform=DswitchTransform(idx);
ModelInfo.t=tout(idx);ModelInfo.tConst=tConst(idx);
ModelInfo.Vo=Vo(idx);ModelInfo.Rload=Rload(idx);
ModelInfo.currentInput=currentInput(idx);
ModelInfo.voltageInput=voltageInput(idx);

ModelInfo.x_fi=[ModelInfo.t ModelInfo.currentInput ModelInfo.Dswitch ModelInfo.Rload 1-ModelInfo.Dswitch ModelInfo.voltageInput];
ModelInfo.y_fi=ModelInfo.Dswitch.*SystemSettings.Vin-(1-ModelInfo.Dswitch).*ModelInfo.Vf;

%% 4-Input signals for x_fv
% idx=sort(randomSeq(3*dataNum+1:4*dataNum));
% idx=idx+5000;
ModelInfo.x_fv_idx=idx_fv;idx=ModelInfo.x_fv_idx;
ModelInfo.IL=IL(idx);ModelInfo.Vf=Vf(idx);
ModelInfo.Dswitch=Dswitch(idx);ModelInfo.DswitchTransform=DswitchTransform(idx);
ModelInfo.t=tout(idx);ModelInfo.tConst=tConst(idx);
ModelInfo.Vo=Vo(idx);ModelInfo.Rload=Rload(idx);
ModelInfo.currentInput=currentInput(idx);
ModelInfo.voltageInput=voltageInput(idx);

ModelInfo.x_fv=[ModelInfo.t ModelInfo.currentInput ModelInfo.Dswitch ModelInfo.Rload 1-ModelInfo.Dswitch ModelInfo.voltageInput];
ModelInfo.y_fv=0*ones(length(ModelInfo.y_fi),1);

%% Testing signals
idx=idx_ui;
% syncErrorLevel=10;% 0(0us) or 10(2us);
syncUpper=floor(syncErrorLevel/100*res);
% arrange addedIdx
addedIdx=randi([0,syncUpper],1,0.5*length(idx));
addedIdx=[addedIdx;addedIdx];
addedIdx=addedIdx(:)';
addedIdx=[addedIdx(2:end) addedIdx(1)];
% add idx disturbance
syncIdx=addedIdx+idx;
% idx=0.3e4:100:2e4;
ModelInfoOriginal.tInitial=tConst;
ModelInfoOriginal.x_ui_idx=idx;ModelInfoOriginal.x_uv_idx=idx;ModelInfoOriginal.x_fi_idx=idx;ModelInfoOriginal.x_fv_idx=idx;
ModelInfoOriginal.IL=IL(idx);ModelInfoOriginal.Vf=Vf(idx);
ModelInfoOriginal.Dswitch=Dswitch(idx);ModelInfoOriginal.DswitchTransform=DswitchTransform(idx);
ModelInfoOriginal.t=tout(idx);ModelInfoOriginal.tConst=tConst(idx);
ModelInfoOriginal.Vo=Vo(syncIdx);
ModelInfoOriginal.Rload=Rload(idx);
ModelInfoOriginal.currentInput=currentInput(idx);
ModelInfoOriginal.voltageInput=voltageInput(idx);

ModelInfoOriginal.x_ui=[ModelInfoOriginal.t ModelInfoOriginal.currentInput ModelInfoOriginal.Dswitch ModelInfoOriginal.Rload 1-ModelInfoOriginal.Dswitch ModelInfoOriginal.voltageInput];
ModelInfoOriginal.x_fi=[ModelInfoOriginal.t ModelInfoOriginal.currentInput ModelInfoOriginal.Dswitch ModelInfoOriginal.Rload 1-ModelInfoOriginal.Dswitch ModelInfoOriginal.voltageInput];
ModelInfoOriginal.x_uv=ModelInfoOriginal.x_ui; ModelInfoOriginal.x_fv=ModelInfoOriginal.x_fi;

ModelInfoOriginal.y_ui=[ModelInfoOriginal.IL];
ModelInfoOriginal.y_fi=ModelInfoOriginal.Dswitch.*SystemSettings.Vin-(1-ModelInfoOriginal.Dswitch).*ModelInfoOriginal.Vf;

ModelInfoOriginal.y_fv=0*ones(length(ModelInfoOriginal.y_fi),1);
ModelInfoOriginal.y_uv=[ModelInfoOriginal.Vo];%+0.12*(5e-2*randn((length(ModelInfoOriginal.Vo)),1));

%% prepare the data for python code
t=ModelInfoOriginal.t;
dt=t(2:2:end);
dt=[dt';dt'];
dt=reshape(dt,[1,numel(dt)])';
DswitchTransform=ModelInfoOriginal.DswitchTransform;%./ModelInfoOriginal.DswitchTransform;
Dswitch=[ModelInfoOriginal.Dswitch(2:end); ModelInfoOriginal.Dswitch(1)] ;
Rload=ModelInfoOriginal.Rload;

CurrentInput=[ModelInfoOriginal.currentInput(2:end); ModelInfoOriginal.currentInput(1)];
VoltageInput=[ModelInfoOriginal.voltageInput(2:end); ModelInfoOriginal.voltageInput(1)];

Vin=24;
Vf=1;

Current=ModelInfoOriginal.y_ui;

Voltage=ModelInfoOriginal.y_uv;

idxD_lower=[find(diff(CurrentInput)~=0);length(CurrentInput)]+1;idxD_lower=idxD_lower(1:end-1);
tLower=t(idxD_lower);
RloadLower=Rload(idxD_lower);
DswitchLower=ModelInfoOriginal.tConst;
CurrentInputLower=Current;
VoltageInputLower=Voltage;

output_file = ['buckSimulation_' num2str(experimentalSettingIdx-1) '.mat'];

save(output_file, 't', 'dt', 'Dswitch',...
    'DswitchTransform', 'Rload', 'CurrentInput',...
    'VoltageInput', 'Current', 'Voltage', 'tLower', 'RloadLower', ...
    'DswitchLower', 'CurrentInputLower', 'VoltageInputLower', 'forwaredBackwaredIndicator', 'res');

