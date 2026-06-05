%% ============================================================
% ChanEst Baseline CNN Channel Estimator
% ============================================================

clear; clc; close all;

%% Load dataset
%load("/MATLAB Drive/ChanEst-Dataset-Generation/data/6G_ChanEst_Dataset_1x1_10000Samples_20260403_235802.mat"'); 

load("/MATLAB Drive/dataset/6G_ChanEst_Dataset_10k_Samples.mat"'); 


% expected variables:
% X_input  [K x L x 2 x N]
% Y_label  [K x L x 2 x N]
% snrLog_dB [N x 1]

[K,L,~,N] = size(X_input);

fprintf("Dataset loaded: %d samples\n",N); %[output:1b99ff19]

%% Train / Validation / Test split
rng(1)

idx = randperm(N);

trainRatio = 0.8;
valRatio   = 0.1;

nTrain = floor(trainRatio*N);
nVal   = floor(valRatio*N);

trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1:nTrain+nVal);
testIdx  = idx(nTrain+nVal+1:end);

XTrain = X_input(:,:,:,trainIdx);
YTrain = Y_label(:,:,:,trainIdx);

XVal = X_input(:,:,:,valIdx);
YVal = Y_label(:,:,:,valIdx);

XTest = X_input(:,:,:,testIdx);
YTest = Y_label(:,:,:,testIdx);

snrTest = snrLog_dB(testIdx);

%% ============================================================
% Baseline CNN Architecture
% ============================================================

layers = [

imageInputLayer([K L 2],"Normalization","none","Name","input")

convolution2dLayer(3,64,"Padding","same","Name","conv1")
reluLayer("Name","relu1")

convolution2dLayer(3,64,"Padding","same","Name","conv2")
reluLayer("Name","relu2")

convolution2dLayer(3,64,"Padding","same","Name","conv3")
reluLayer("Name","relu3")

convolution2dLayer(3,2,"Padding","same","Name","conv_out")

regressionLayer("Name","output")

];

%% Training options

options = trainingOptions("adam", ...
    MaxEpochs= 20, ...
    MiniBatchSize=16, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal,YVal}, ...
    ValidationFrequency=100, ...
    Verbose=true, ...
    Plots="training-progress");

%% Train CNN

fprintf("Training CNN baseline...\n") %[output:134c42df]

net = trainNetwork(XTrain,YTrain,layers,options); %[output:08ec353c] %[output:984a3496]

%% ============================================================
% CNN Prediction
% ============================================================

YPred = predict(net,XTest);

%% ============================================================
% Compute NMSE
% ============================================================

nmseCNN = zeros(numel(testIdx),1);
nmseLS  = zeros(numel(testIdx),1);

for i = 1:numel(testIdx)

    Htrue = YTest(:,:,:,i);
    Hls   = XTest(:,:,:,i);
    Hcnn  = YPred(:,:,:,i);

    nmseLS(i) = norm(Hls(:)-Htrue(:))^2 / norm(Htrue(:))^2;
    nmseCNN(i) = norm(Hcnn(:)-Htrue(:))^2 / norm(Htrue(:))^2;

end

%% ============================================================
% NMSE vs SNR
% ============================================================

snrBins = -10:5:30;
nmseLS_bin = zeros(length(snrBins)-1,1);
nmseCNN_bin = zeros(length(snrBins)-1,1);

for b = 1:length(snrBins)-1

    idx = snrTest>=snrBins(b) & snrTest<snrBins(b+1);

    nmseLS_bin(b)  = mean(nmseLS(idx));
    nmseCNN_bin(b) = mean(nmseCNN(idx));

end

snrAxis = snrBins(1:end-1)+2.5;

%% ============================================================
% Plot NMSE vs SNR
% ============================================================

figure %[output:7cd4edc0]
semilogy(snrAxis,nmseLS_bin,'o-','LineWidth',2) %[output:7cd4edc0]
hold on %[output:7cd4edc0]
semilogy(snrAxis,nmseCNN_bin,'s-','LineWidth',2) %[output:7cd4edc0]

grid on %[output:7cd4edc0]
xlabel('SNR (dB)') %[output:7cd4edc0]
ylabel('NMSE') %[output:7cd4edc0]
legend('LS Baseline','CNN Estimator') %[output:7cd4edc0]
title('Channel Estimation Performance on ChanEst Dataset') %[output:7cd4edc0]

%% Save figure

saveas(gcf,'ChanEst_CNN_vs_LS.png') %[output:7cd4edc0]

fprintf("Evaluation complete.\n") %[output:133814e6]


%% ============================================================
% NMSE vs Delay Spread
% ============================================================

delayTest = delaySpreadLog_s(testIdx);

delayBins = linspace(min(delayTest), max(delayTest), 8);

nmseLS_delay = zeros(length(delayBins)-1,1);
nmseCNN_delay = zeros(length(delayBins)-1,1);

for b = 1:length(delayBins)-1

    idx = delayTest>=delayBins(b) & delayTest<delayBins(b+1);

    nmseLS_delay(b)  = mean(nmseLS(idx));
    nmseCNN_delay(b) = mean(nmseCNN(idx));

end

delayAxis = (delayBins(1:end-1)+delayBins(2:end))/2;

figure %[output:5457e019]
plot(delayAxis*1e9,nmseLS_delay,'o-','LineWidth',2) %[output:5457e019]
hold on %[output:5457e019]
plot(delayAxis*1e9,nmseCNN_delay,'s-','LineWidth',2) %[output:5457e019]

grid on %[output:5457e019]
xlabel('Delay Spread (ns)') %[output:5457e019]
ylabel('NMSE') %[output:5457e019]
legend('LS Baseline','CNN Estimator') %[output:5457e019]
title('Channel Estimation Performance vs Delay Spread') %[output:5457e019]



%% ============================================================
% NMSE vs Doppler
% ============================================================

dopplerTest = dopplerLog_Hz(testIdx);

dopplerBins = linspace(min(dopplerTest), max(dopplerTest), 8);

nmseLS_dopp = zeros(length(dopplerBins)-1,1);
nmseCNN_dopp = zeros(length(dopplerBins)-1,1);

for b = 1:length(dopplerBins)-1

    idx = dopplerTest>=dopplerBins(b) & dopplerTest<dopplerBins(b+1);

    nmseLS_dopp(b)  = mean(nmseLS(idx));
    nmseCNN_dopp(b) = mean(nmseCNN(idx));

end

dopplerAxis = (dopplerBins(1:end-1)+dopplerBins(2:end))/2;

figure %[output:5fdacd36]
plot(dopplerAxis,nmseLS_dopp,'o-','LineWidth',2) %[output:5fdacd36]
hold on %[output:5fdacd36]
plot(dopplerAxis,nmseCNN_dopp,'s-','LineWidth',2) %[output:5fdacd36]

grid on %[output:5fdacd36]
xlabel('Doppler Shift (Hz)') %[output:5fdacd36]
ylabel('NMSE') %[output:5fdacd36]
legend('LS Baseline','CNN Estimator') %[output:5fdacd36]
title('Channel Estimation Performance vs Doppler') %[output:5fdacd36]



