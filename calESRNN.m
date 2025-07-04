

stim1 = 10;
stim1Off = 100;
stim2 = 160;
bound = 2.58;


load('~/code/lowrank_inference/notebooks/trajVanilla.mat');

cfdTraj = double(permute(cfdTraj, [1,3,2]));
tfTraj = double(permute(tfTraj, [1,3,2]));

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


[cfdDirMod, cfdDirES] = calMod(cfdTraj, cfdLeft, cfdRight, bound);
[cfdColMod, cfdColES] = calMod(cfdTraj, cfdRed, cfdGreen, bound);
[cfdCxtMod, cfdCxtES] = calMod(cfdTraj, cfdCxt1, cfdCxt2, bound);

[tfDirMod, tfDirES] = calMod(tfTraj, tfLeft, tfRight, bound);
[tfColMod, tfColES] = calMod(tfTraj, tfRed, tfGreen, bound);
[tfCxtMod, tfCxtES] = calMod(tfTraj, tfCxt1, tfCxt2, bound);

%% plot psth 
cfdRL = cfdTraj(cfdDecision == -1& cfdCoh > 0,:,:);
cfdRR = cfdTraj(cfdDecision == 1& cfdCoh > 0,:,:);
cfdGL = cfdTraj(cfdDecision == -1& cfdCoh < 0,:,:);
cfdGR = cfdTraj(cfdDecision == 1 & cfdCoh < 0,:,:);


tfRL = tfTraj(tfDecision == -1& tfCoh > 0,:,:);
tfRR = tfTraj(tfDecision == 1& tfCoh > 0,:,:);
tfGL = tfTraj(tfDecision == -1& tfCoh < 0,:,:);
tfGR = tfTraj(tfDecision == 1 & tfCoh < 0,:,:);


for n = 100:110
    figure; 
    subplot(1,2,1), hold on
    plot(mean(squeeze(cfdRL(:,n,:)),1),'r-')
    plot(mean(squeeze(cfdRR(:,n,:)),1),'r--')
    plot(mean(squeeze(cfdGL(:,n,:)),1),'g-')
    plot(mean(squeeze(cfdGR(:,n,:)),1),'g--')
    
    subplot(1,2,2),hold on
    plot(mean(squeeze(tfRL(:,n,:)),1),'r-')
    plot(mean(squeeze(tfRR(:,n,:)),1),'r--')
    plot(mean(squeeze(tfGL(:,n,:)),1),'g-')
    plot(mean(squeeze(tfGR(:,n,:)),1),'g--')
    
    pause;
    close
end

%% 

% 
% tfColES = tfColES(81:end,:);
% cfdColES = cfdColES(81:end,:);
% 
% tfCxtES = tfCxtES(81:end,:);
% cfdCxtES = cfdCxtES(81:end,:);
% 
% tfDirES = tfDirES(81:end,:);
% cfdDirES = cfdDirES(81:end,:);

figure; hold on
plot(max(tfDirES,[],2), max(cfdDirES,[],2), 'k.', 'markersize', 5)
title('dir')
ylim([0,4])
xlim([0,4])


figure; hold on
plot(max(tfColES,[],2), max(cfdColES,[],2), 'k.', 'markersize', 5)
title('col')
ylim([0,10])
xlim([0,10])


figure; hold on
plot(max(tfCxtES,[],2), max(cfdCxtES,[],2), 'k.', 'markersize', 5)
title('cxt')
ylim([0,5])
xlim([0,35])


figure; hold on
plot(max(tfCxtES,[],2), max(cfdColES,[],2), 'k.', 'markersize', 5)
title('tfCxt-cfdCol')
ylim([0,35])
xlim([0,35])

figure; hold on
plot(max(tfColES,[],2), max(cfdCxtES,[],2), 'k.', 'markersize', 5)
title('tfCol-cfdCxt')
ylim([0,35])
xlim([0,35])
