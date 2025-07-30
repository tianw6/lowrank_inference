clear; close all; clc


fr = load('4AreasA5_0.mat').firingRatesAverage;


for ii = 40:50

    figure; hold on
    
    plot(squeeze(fr(ii,1,1,:)), 'r-');
    plot(squeeze(fr(ii,1,2,:)), 'r--');
    plot(squeeze(fr(ii,2,1,:)), 'g-');
    plot(squeeze(fr(ii,2,2,:)), 'g--');
end

%%
for ii = 340:350

    figure; hold on
    
    plot(squeeze(fr(ii,1,1,:)), 'r-');
    plot(squeeze(fr(ii,1,2,:)), 'r--');
    plot(squeeze(fr(ii,2,1,:)), 'g-');
    plot(squeeze(fr(ii,2,2,:)), 'g--');
end