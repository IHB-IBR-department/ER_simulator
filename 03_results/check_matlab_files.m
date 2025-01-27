clear
clc

load('task.mat')
load('task_microtime.mat')

%%
BOLD_MTv2 = downsample(BOLD',400/16)';

figure(1)
subplot(3,1,1);
plot(BOLD');
subplot(3,1,2);
plot(BOLD_MTv2');
subplot(3,1,3);
plot(BOLD_MT');

figure(2)
plot(BOLD_MT' - BOLD_MTv2');

%%
BOLD_MTv3 = downsample(BOLD',400)';

figure(3)
plot(BOLD_TR' - BOLD_MTv3');

%%
syn_act_MTv2 = downsample(syn_act',400/16)';

figure(4)
subplot(2,1,1);
plot(syn_act');
subplot(2,1,2);
plot(syn_act_MTv2');

figure(5)
plot(syn_act_MT' - syn_act_MTv2');

