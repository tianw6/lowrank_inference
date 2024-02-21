function [modulation,effectSize] = calMod(trials, group1, group2, bound)
% calculate whether 99%ile bound overlap and effect size 
    
    modulation = [];
    effectSize = [];

    group1Trials = trials(group1,:,:);
    group2Trials = trials(group2,:,:);


    for n = 1:size(group1Trials,2)

        a = squeeze(group1Trials(:,n,:));
        b = squeeze(group2Trials(:,n,:));

        a_sem = std(a)./sqrt(size(a,1));
        b_sem = std(b)./sqrt(size(b,1));

        a_stat = [mean(a,1) - a_sem.*bound; mean(a,1) + a_sem.*bound];
        b_stat = [mean(b,1) - b_sem.*bound; mean(b,1) + b_sem.*bound];

        effect_size_raw = abs(mean(a,1) - mean(b,1))./sqrt(0.5.*(var(a) + var(b)));

        modulation1 = a_stat(1,:) > b_stat(2,:) | a_stat(2,:) < b_stat(1,:);
%         effectSize1 = effect_size_raw.*modulation1;
        
        modulation(n,:) = modulation1;
        effectSize(n,:) = effect_size_raw;
    end
    
end

