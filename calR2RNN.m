clear; clc


% pre-process saved RNN FR for regression
stim1 = 10;
stim1Off = 100;
stim2 = 160;
bound = 2.58;


load('~/code/lowrank_inference/notebooks/trajTotalSep.mat');



cfdTraj = double(permute(cfdTraj, [3,2,1]));
tfTraj = double(permute(tfTraj, [3,2,1]));

cfdLeft = cfdDecision == -1;
cfdRight = cfdDecision == 1;
cfdRed = cfdCoh > 0;
cfdGreen = cfdCoh < 0;
cfdCxt1 = (cfdCoh > 0 & cfdDecision == -1) | (cfdCoh < 0 & cfdDecision == 1);
cfdCxt2 = (cfdCoh > 0 & cfdDecision == 1) | (cfdCoh < 0 & cfdDecision == -1);

tfRed = tfCoh > 0;
tfGreen = tfCoh < 0;
tfCxt1 = (tfCoh > 0 & tfDecision == -1) | (tfCoh < 0 & tfDecision == 1);
tfCxt2 = (tfCoh > 0 & tfDecision == 1) | (tfCoh < 0 & tfDecision == -1);
tfLeft = tfDecision == -1;
tfRight = tfDecision == 1;



TF = struct;
TF.FR = tfTraj;
TF.time = -100:10:2000;
TF.left = tfLeft;
TF.red = tfRed;
TF.cxt1 = tfCxt1;

CMod = TF;
% save('~/Desktop/TFRNN.mat', 'CMod');
%%
CFD = struct;
CFD.FR = cfdTraj;
CFD.time = -100:10:2000;
CFD.left = cfdLeft;
CFD.red = cfdRed;
CFD.cxt1 = cfdCxt1;

CMod = CFD;
% save('~/Desktop/CFDRNN.mat', 'CMod');



%%
% load a dataset
load('~/Desktop/CFDRNN.mat')

dataSet = 'CFDT'; 

switch(dataSet)
    case {'CFDC', 'TFT'}
    resultsStruct= struct('R_squared', [], 'model', [], 'regressor', []);

    case {'CFDT', 'TFC'}
        resultsStructColor = struct('R_squared', [], 'model', [], 'regressor', []);
        resultsStructCxt = struct('R_squared', [], 'model', [], 'regressor', []);
        resultsStructSide = struct('R_squared', [], 'model', [], 'regressor', []);
end 


for ii = 1:length(CMod)
    data = CMod(ii).FR; 
    color = CMod(ii).red'; 
    cxt = CMod(ii).cxt1'; 
    side = CMod(ii).left'; 

    numUnits = size(data, 1);
    numTimepoints = size(data, 2); 
    numTrials = size(data, 3); 

    timeSteps = 1:numTimepoints(end);

    allNeuralValues = struct('unitValues', cell(1, numUnits));

    for unitIndex = 1:numUnits
        unitValues = squeeze(data(unitIndex,:,:))';
        allNeuralValues(unitIndex).unitValues = unitValues;
    end

    
    %% regression per unit for each time bin 

    numBin = length(timeSteps); 

    results = zeros(numUnits, numBin); 
    resultsColor  = zeros(numUnits, numBin);
    resultsCxt  = zeros(numUnits, numBin);
    resultsSide  = zeros(numUnits, numBin);

    switch(dataSet)
        case {'CFDC', 'TFT'}
            for l = 1:numUnits
                for i = 1:numBin
                    y = allNeuralValues(l).unitValues(:, i); 
        
                    model = fitlm(color,y); 
                    results(l, i) = model.Rsquared.Ordinary;
                end 
        
            end
        
            % store results 
            resultsStruct(ii).R_squared = results;
            resultsStruct(ii).model = model;
            resultsStruct(ii).regressor = 'color'; 
            resultsStruct(ii).inputData = dataSet; 


        case {'CFDT', 'TFC'}
            for l = 1:numUnits
                for i = 1:numBin
                    y = allNeuralValues(l).unitValues(:, i); 
    
                    modelColor = fitlm(color,y); 
                    resultsColor(l, i) = modelColor.Rsquared.Ordinary;
    
                    modelCxt = fitlm(cxt,y);
                    resultsCxt(l, i) = modelCxt.Rsquared.Ordinary;
    
                    modelSide = fitlm(side,y);
                    resultsSide(l, i) = modelSide.Rsquared.Ordinary;
    
                end 
                fprintf('Unit %d finished \n', l)
            end
    
            % store results 
            resultsStructColor(ii).R_squared = resultsColor;
            resultsStructColor(ii).model = modelColor;
            resultsStructColor(ii).regressor = 'Color'; 
            resultsStructColor(ii).inputData = dataSet; 

            resultsStructCxt(ii).R_squared = resultsCxt;
            resultsStructCxt(ii).model = modelCxt;
            resultsStructCxt(ii).regressor = 'Cxt'; 
            resultsStructCxt(ii).inputData = dataSet; 

            resultsStructSide(ii).R_squared = resultsSide;
            resultsStructSide(ii).model = modelSide;
            resultsStructSide(ii).regressor = 'Side'; 
            resultsStructSide(ii).inputData = dataSet; 
            
            % append all 
            resultsStruct = [resultsStructColor, resultsStructCxt, resultsStructSide]; 

    end

end


%% plot the results 

% dataSet = 'TFC'; 
% load resultsStructTFT.mat

switch(dataSet)
    case {'CFDC', 'TFT'}

    for ii = 1:length(resultsStruct)
        figure; 
        title('color')
        hold on 
        for i = 1:size(resultsStruct(ii).R_squared)
            plot(resultsStruct(ii).R_squared(i, :))
        end 
    
        pause; 
        close; 
    end 
        
    case {'CFDT', 'TFC'}
     
   for ii = 1:length(resultsStruct)
        figure; 
        title(resultsStruct(ii).regressor)
        hold on 
        for i = 1:size(resultsStruct(ii).R_squared)
            plot(resultsStruct(ii).R_squared(i, :))
        end 

        pause; 
        close; 
    end 

end 



%% plot max regression value 
TFC = load('~/Desktop/resultsStructTFRNN.mat').resultsStruct;
CFDT = load('~/Desktop/resultsStructCFDRNN.mat').resultsStruct;

TFcolor = TFC(1).R_squared;
TFcolor = TFcolor(1:60,:);

TFcxt = TFC(2).R_squared;
TFcxt = TFcxt(1:60,:);

TFdir = TFC(3).R_squared;
TFdir = TFdir(1:60,:);


CFDcolor = CFDT(1).R_squared;
CFDcolor = CFDcolor(1:60,:);
CFDcxt = CFDT(2).R_squared;
CFDcxt = CFDcxt(1:60,:);

CFDdir = CFDT(3).R_squared;
CFDdir = CFDdir(1:60,:);

figure; hold on
plot(max(TFcxt,[],2), max(CFDcxt,[],2), 'k.');
xlim([0,1]); ylim([0,1])
title('cxt')
% print('-painters', '-depsc', '~/Desktop/rnnCxtR2.eps', '-r300')

figure; hold on
plot(max(TFcolor,[],2), max(CFDcolor,[],2), 'k.');
xlim([0,1]); ylim([0,1])
title('color')
% print('-painters', '-depsc', '~/Desktop/rnnColR2.eps', '-r300')

figure; hold on
plot(max(TFdir,[],2), max(CFDdir,[],2), 'k.');
xlim([0,1]); ylim([0,1])
title('dir')
% print('-painters', '-depsc', '~/Desktop/rnnDirR2.eps', '-r300')


figure; hold on
plot(max(TFcxt,[],2), max(CFDcolor,[],2), 'k.');
xlim([0,1]); ylim([0,1])
title('TFcxt-CFDcol')

figure; hold on
plot(max(TFcolor,[],2), max(CFDcxt,[],2), 'k.');
xlim([0,1]); ylim([0,1])
title('TFcolor-CFDcxt')
%%
% save('~/Desktop/resultsStructCFDRNN.mat','resultsStruct')