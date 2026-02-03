% ============================================================
% File: Generate_6G_CE_Dataset.m
%
% Purpose:
%   Public, reproducible dataset generation for DL-based channel estimation.
%   "General coverage" by default (stratified sampling across ranges),
%   fully reconfigurable (SISO/MIMO, N, SNR/DS/Doppler, profiles, etc.).
%
% Outputs (core):
%   X_input : Practical channel estimate on full grid (DM-RS -> LS/CDM -> 2D interp)
%   Y_label : Perfect channel frequency response on full grid (ofdm-response)
%
% Tensor format (real/imag packing), Size: K x L x (2*NRx*NTx) x N     (single)
%
% Packing rule:
%   linkIndex  = (tx-1)*NRx + rx        % 1..(NRx*NTx)
%   Re channel = 2*(linkIndex-1)+1
%   Im channel = Re+1
%
% Channel:
%   nrTDLChannel with ChannelResponseOutput="ofdm-response"
%
% Saving:
%   - MAT (v7.3) : MATLAB users
%   - H5         : Python users
%
% Notes:
%   - Normalization is NOT stored; do it during training if needed.
% ============================================================


%%%%%%%%            cfg.overwriteFiles = false

clear; clc; close all;

%% ========================= PORTABLE PROJECT PATH =========================
% Save into <repo>/data relative to THIS script location.
scriptFullPath = mfilename("fullpath");
if strlength(string(scriptFullPath)) == 0
    scriptFolder = pwd;
else
    scriptFolder = fileparts(scriptFullPath);          % expected: <repo>/src
end
projectRoot = fileparts(scriptFolder);                 % expected: <repo>
dataFolder  = fullfile(projectRoot,"data");            % expected: <repo>/data
if ~exist(dataFolder,"dir"); mkdir(dataFolder); end

%% ============================== USER CONFIG ==============================
cfg = struct();

% ---- Dataset size / reproducibility ----
cfg.numSamples   = 5;      % Reconfigurable. 10 for quick test, 10k for release
cfg.randomSeed   = 42;
cfg.showProgress = true;

% ---- Antennas (SISO or MIMO (2x2, 4x4)) ----
cfg.numTxAntennas = 1;      % NTx
cfg.numRxAntennas = 1;      % NRx

% ---- Carrier frequency (only for speed<->doppler relation) ----
cfg.carrierFrequencyHz = 7e9;    % FR3 UPPER BAND (used only to compute speed from Doppler)

% ---- Scenario ranges (general coverage, low->high) ----
cfg.delayProfiles        = ["TDL-A","TDL-B","TDL-C","TDL-D","TDL-E"];
cfg.snrRange_dB          = [-10 30];              % dB
cfg.delaySpreadRange_s   = [10 2000]*1e-9;        % 10 ns .. 2000 ns

% Doppler selection:
%   "direct": sample f_d uniformly in cfg.DopplerRange_Hz, then compute implied speed
%   "speed" : sample speed uniformly in cfg.speedRange_mps, then compute f_d
cfg.dopplerMode     = "direct";
cfg.DopplerRange_Hz = [5 5000];   % set [0 ...] if you want perfectly static samples too
cfg.speedRange_mps  = [1 120];    % used only if dopplerMode="speed"

% Optional MIMO correlation sets (only used if NTx>1 or NRx>1)
cfg.mimoCorrelationSet = ["Low","Medium","High"];

% ---- Sampling style ----------------
cfg.paramSampling = "stratified";   % "stratified" or "rand"

% ---- OFDM demod tuning ----
cfg.cyclicPrefixFraction = 0.55;

% ---- Saving ----
cfg.saveToDisk     = true;
cfg.saveMat        = true;
cfg.saveH5         = true;
cfg.overwriteFiles = false;     % if false -> adds timestamp suffix
cfg.h5DeflateLevel = 4;         % Apply moderate compression         

% ---- Memory mode ----
% "stream" writes sample-by-sample (recommended for large N or MIMO)
% "memory" keeps tensors in RAM (fine for small N or SISO)
cfg.writeMode = "stream";      % "stream" or "memory"

% ---- Robustness ----
cfg.maxChannelAttempts = 3;   % retry channel call with seed tweaks if rare internal failure occurs
cfg.verboseWarnings    = false; % set true to print warnings/fallback messages during generation

% ---- Timing offset handling ----
% "circshift" preserves length (safer for demod), "truncate" discards leading samples
cfg.timingOffsetMode = "circshift";   % "circshift" | "truncate"

% ---- Quick QA ----
cfg.doQuickQA      = true;    % prints summary + light stats
cfg.doPDPQuickLook = true;    % 1-sample effective PDP sanity-check

% Output folder and dataset naming
cfg.outputFolder = dataFolder;
cfg.baseName = sprintf("6G_ChanEst_Dataset_%dx%d_%dSamples", ...
    cfg.numTxAntennas, cfg.numRxAntennas, cfg.numSamples);

cfg.datasetName  = "ChanEst Dataset: A Reproducible and Reconfigurable 6G Channel Estimation Dataset";
cfg.datasetVer   = "v1.0";
cfg.authorName   = "Obinna Okoyeigbo";

%% ============================== ASSERTS ==================================
assert(cfg.numSamples >= 1, "cfg.numSamples must be >= 1");
assert(cfg.snrRange_dB(2) > cfg.snrRange_dB(1), "Invalid SNR range.");
assert(cfg.delaySpreadRange_s(1) >= 0 && cfg.delaySpreadRange_s(2) > cfg.delaySpreadRange_s(1), ...
    "Invalid delay spread range.");
assert(cfg.carrierFrequencyHz > 0, "carrierFrequencyHz must be positive.");

if cfg.dopplerMode == "direct"
    assert(cfg.DopplerRange_Hz(1) >= 0 && cfg.DopplerRange_Hz(2) > cfg.DopplerRange_Hz(1), ...
        "Invalid Doppler range.");
end

if cfg.writeMode == "stream" && ~cfg.saveToDisk
    error('writeMode="stream" requires saveToDisk=true (otherwise nothing is stored).');
end

%% ============================== RNG ======================================
rng(cfg.randomSeed,"twister");

%% ================== NR CARRIER / PDSCH / DM-RS ===========================
carrier = nrCarrierConfig;
carrier.NSizeGrid         = 51;      % 51 RB -> 612 subcarriers
carrier.SubcarrierSpacing = 60;      % kHz
carrier.CyclicPrefix      = "Normal";
carrier.NCellID           = 2;

ofdmInfo = nrOFDMInfo(carrier);

pdsch = nrPDSCHConfig;
pdsch.PRBSet           = 0:carrier.NSizeGrid-1;
pdsch.SymbolAllocation = [0 carrier.SymbolsPerSlot];   % full slot
pdsch.MappingType      = "A";
pdsch.NID              = carrier.NCellID;
pdsch.RNTI             = 1;

% Simple rule: treat each Tx antenna as one "layer"
pdsch.NumLayers = cfg.numTxAntennas;

% DM-RS config
pdsch.DMRS.DMRSTypeAPosition       = 2;
pdsch.DMRS.DMRSLength              = 1;
pdsch.DMRS.DMRSAdditionalPosition  = 1;
pdsch.DMRS.DMRSConfigurationType   = 2;   % Type-2 (CDM possible)
pdsch.DMRS.NumCDMGroupsWithoutData = 1;
pdsch.DMRS.DMRSPortSet = 0:(pdsch.NumLayers-1);

% Generate DM-RS
dmrsSymbols = nrPDSCHDMRS(carrier, pdsch, 'OutputDataType','single');
dmrsIndices = nrPDSCHDMRSIndices(carrier, pdsch);
dmrsSubs    = nrPDSCHDMRSIndices(carrier, pdsch, 'IndexStyle','subscript'); % [k l portIndex]

numPorts = pdsch.NumLayers;

K = carrier.NSizeGrid*12;
L = carrier.SymbolsPerSlot;

% Pilot positions per port (used by our interpolator)
pilotSubc = cell(numPorts,1);
pilotSym  = cell(numPorts,1);
for p = 1:numPorts
    rowsP = (double(dmrsSubs(:,3)) == p);
    pilotSubc{p} = double(dmrsSubs(rowsP,1));
    pilotSym{p}  = double(dmrsSubs(rowsP,2));
end

% Pilot density metadata (time-frequency overhead)
dmrsPos2D = unique(double(dmrsSubs(:,1:2)),"rows");
pilotRE_unique = size(dmrsPos2D,1);
pilotDensity_unique = pilotRE_unique / (K*L);

pilotRE_perPort = zeros(numPorts,1);
for p = 1:numPorts
    pilotRE_perPort(p) = numel(pilotSubc{p});
end
pilotDensity_perPort = pilotRE_perPort / (K*L);

%% =================== TX GRID / WAVEFORM (DM-RS ONLY) =====================
% For Type-2 DM-RS with multi-layer, avoid any "txGrid(linInd)=..."


txGrid = nrResourceGrid(carrier, pdsch.NumLayers);  % K x L x NumLayers

for idx = 1:numel(dmrsSymbols)
    k = dmrsSubs(idx,1);
    l = dmrsSubs(idx,2);
    p = dmrsSubs(idx,3);
    txGrid(k,l,p) = dmrsSymbols(idx);
end

% OFDM modulate DM-RS-only grid
txWaveformBase = nrOFDMModulate(carrier, txGrid);

%% ======================= CHANNEL MODEL ===================================
tdl = nrTDLChannel;
tdl.NumTransmitAntennas   = cfg.numTxAntennas;
tdl.NumReceiveAntennas    = cfg.numRxAntennas;
tdl.SampleRate            = ofdmInfo.SampleRate;
tdl.ChannelResponseOutput = "ofdm-response";

% Compute worst-case padding once (avoids calling info(tdl) every sample)
maxChDelayWorst = 0;
tdl.DelaySpread = max(cfg.delaySpreadRange_s);

for ii = 1:numel(cfg.delayProfiles)
    release(tdl);
    tdl.DelayProfile = cfg.delayProfiles(ii);
    chInfoNow = info(tdl);
    maxChDelayWorst = max(maxChDelayWorst, chInfoNow.MaximumChannelDelay);
end

padLen = maxChDelayWorst + ofdmInfo.Nfft;   % one FFT as safety
release(tdl);

%% ===================== PARAMETER PLAN (EVEN COVERAGE) =====================
N = cfg.numSamples;

u1 = localStratified01(N);
u2 = localStratified01(N);
u3 = localStratified01(N);

if cfg.paramSampling == "rand"
    u1 = rand(N,1); u2 = rand(N,1); u3 = rand(N,1);
end

% Shuffle each dimension so SNR/DS/Doppler aren't artificially correlated
uSNR = u1(randperm(N));
uDS  = u2(randperm(N));
uFD  = u3(randperm(N));

snrPlan_dB = cfg.snrRange_dB(1) + uSNR * diff(cfg.snrRange_dB);
dsPlan_s   = cfg.delaySpreadRange_s(1) + uDS  * diff(cfg.delaySpreadRange_s);
fdPlan_Hz  = cfg.DopplerRange_Hz(1) + uFD     * diff(cfg.DopplerRange_Hz);

% Balanced delay profile assignment
profList = repmat(cfg.delayProfiles(:), ceil(N/numel(cfg.delayProfiles)), 1);
profPlan = profList(1:N);
profPlan = profPlan(randperm(N));

% Balanced MIMO correlation assignment (only matters for MIMO)
corrList = repmat(cfg.mimoCorrelationSet(:), ceil(N/numel(cfg.mimoCorrelationSet)), 1);
corrPlan = corrList(1:N);
corrPlan = corrPlan(randperm(N));

% Pre-generate per-sample seeds (reproducible)
seedPlan = uint32(randi([1 2^31-2], N, 1));

%% ===================== OUTPUT / STREAM SETUP =============================
numLinks    = cfg.numRxAntennas * cfg.numTxAntennas;
numChannels = 2 * numLinks;

% Output filenames
baseNameFinal = cfg.baseName;
if cfg.saveToDisk && ~cfg.overwriteFiles
    tstamp = datetime("now","Format","yyyyMMdd_HHmmss");
    baseNameFinal = cfg.baseName + "_" + string(tstamp);
end

matPath = fullfile(cfg.outputFolder, baseNameFinal + ".mat");
h5Path  = fullfile(cfg.outputFolder, baseNameFinal + ".h5");

% Always log scenario params in RAM (small)
delaySpreadLog_s = zeros(N,1);
speedLog_mps     = zeros(N,1);
dopplerLog_Hz    = zeros(N,1);
snrLog_dB        = zeros(N,1);
seedLog          = zeros(N,1,"uint32");
profileLog       = strings(N,1);
mimoCorrLog      = strings(N,1);

% Quick QA logs (computed on-the-fly; works even in stream mode)
corrH11Log = zeros(N,1,'single');
nmseH11Log = zeros(N,1,'single');

% Allocate tensors only in memory mode
if cfg.writeMode == "memory"
    X_input = zeros(K,L,numChannels,N,"single");
    Y_label = zeros(K,L,numChannels,N,"single");
else
    X_input = [];   
    Y_label = [];   
end

% Prepare output folder
if cfg.saveToDisk && ~exist(cfg.outputFolder,"dir")
    mkdir(cfg.outputFolder);
end

% Streaming writers
useMatStream = cfg.saveToDisk && cfg.saveMat && (cfg.writeMode == "stream");
useH5Stream  = cfg.saveToDisk && cfg.saveH5  && (cfg.writeMode == "stream");

% MAT streaming: create v7.3 file and pre-size variables
if useMatStream
    if exist(matPath,"file") && cfg.overwriteFiles
        delete(matPath);
    end
    mf = matfile(matPath,"Writable",true);
    mf.X_input(K,L,numChannels,N) = single(0);
    mf.Y_label(K,L,numChannels,N) = single(0);
end

% H5 streaming: create datasets once, then write slices
if useH5Stream
    if exist(h5Path,"file") && cfg.overwriteFiles
        delete(h5Path);
    end
    localH5Create4D(h5Path, "/X_input", [K L numChannels N], cfg.h5DeflateLevel);
    localH5Create4D(h5Path, "/Y_label", [K L numChannels N], cfg.h5DeflateLevel);
end

%% =================== CONSTANTS (OUTSIDE LOOP) =============================
c0 = 299792458;

exampleIdx = [];
exampleHgt = [];

% CDM lengths hint for channel estimation (robust default)
% Type-2 DM-RS commonly uses frequency CDM length 2.
if pdsch.DMRS.DMRSConfigurationType == 2
    if pdsch.DMRS.DMRSLength == 2
        cdmLengths = [2 2];
    else
        cdmLengths = [2 1];
    end
else
    cdmLengths = [1 1];
end

%% ============================ MAIN LOOP ===================================
for n = 1:N

    % Planned scenario parameters
    snrNow_dB = snrPlan_dB(n);
    dsNow     = dsPlan_s(n);
    profNow   = profPlan(n);

    switch cfg.dopplerMode
        case "direct"
            fdNow = fdPlan_Hz(n);
            vNow  = fdNow * c0 / cfg.carrierFrequencyHz;
        case "speed"
            vNow  = cfg.speedRange_mps(1) + rand * diff(cfg.speedRange_mps);
            fdNow = abs(vNow) * cfg.carrierFrequencyHz / c0;
        otherwise
            error('cfg.dopplerMode must be "direct" or "speed".');
    end

    % Logs
    snrLog_dB(n)        = snrNow_dB;
    delaySpreadLog_s(n) = dsNow;
    dopplerLog_Hz(n)    = fdNow;
    speedLog_mps(n)     = vNow;
    profileLog(n)       = profNow;
    seedLog(n)          = seedPlan(n);

    if (cfg.numTxAntennas > 1) || (cfg.numRxAntennas > 1)
        mimoCorrLog(n) = corrPlan(n);
    else
        mimoCorrLog(n) = "SISO";
    end

    % Pad waveform once per sample (captures multipath tail safely)
    txWaveform = [txWaveformBase; zeros(padLen,cfg.numTxAntennas,"like",txWaveformBase)];

    % Generate channel (retry on rare failures)
    success = false;
    for attempt = 1:cfg.maxChannelAttempts

        release(tdl);
        tdl.DelayProfile        = profNow;
        tdl.DelaySpread         = dsNow;
        tdl.MaximumDopplerShift = fdNow;

        if (cfg.numTxAntennas > 1) || (cfg.numRxAntennas > 1)
            tdl.MIMOCorrelation = mimoCorrLog(n);
        end

        seedTry = double(mod(double(seedPlan(n)) + (attempt-1), 2^31-1));
        tdl.Seed = seedTry;

        try
            [rxWaveform, H_perfect, timingOffset] = tdl(txWaveform, carrier);
            success = true;
            break;
        catch ME
            if attempt == cfg.maxChannelAttempts
                rethrow(ME);
            end
        end
    end

    if ~success
        error("Channel generation failed unexpectedly at sample %d.", n);
    end

    % Add AWGN (noise per Rx antenna; Nfft scaling used for OFDM-consistent variance)
    snrLin  = 10^(snrNow_dB/10);
    noiseStd = 1/sqrt(2*ofdmInfo.Nfft*snrLin);
    noiseStd = cast(noiseStd,"like",real(rxWaveform));

    noiseRe = randn(size(rxWaveform),"like",real(rxWaveform));
    noiseIm = randn(size(rxWaveform),"like",real(rxWaveform));
    rxWaveform = rxWaveform + noiseStd * complex(noiseRe,noiseIm);

    % Timing alignment (robust; keeps length stable for demod)
    if timingOffset > 0
        if timingOffset < size(rxWaveform,1)
            if cfg.timingOffsetMode == "truncate"
                rxWaveform = rxWaveform(1+timingOffset:end,:);
            else
                rxWaveform = circshift(rxWaveform, -timingOffset, 1);
                rxWaveform(end-timingOffset+1:end,:) = 0;
            end
        else
            if cfg.verboseWarnings
                warning('Sample %d: timingOffset=%d exceeds waveform length=%d', ...
                    n, timingOffset, size(rxWaveform,1));
            end
            rxWaveform = zeros(size(rxWaveform),"like",rxWaveform);
        end
    end

    % OFDM demod
    rxGrid = nrOFDMDemodulate(carrier, rxWaveform, 'CyclicPrefixFraction', cfg.cyclicPrefixFraction);
    rxGrid = localFixRxGridSize(rxGrid, K, L);

    % ------------------------------------------------------------
    % Pilot-domain channel estimation (robust across releases)
    %   Try: (carrier, rxGrid, dmrsIndices, dmrsSymbols, 'CDMLengths',..., 'Interpolation','off')
    %   Fallback: same without 'Interpolation'
    %   Final: no CDMLengths call form
    % ------------------------------------------------------------
    gotPilotOnly = false;

    try
        H_pilots = nrChannelEstimate(carrier, rxGrid, dmrsIndices, dmrsSymbols, ...
            'CDMLengths', cdmLengths, ...
            'AveragingWindow', [1 1], ...
            'Interpolation', 'off');
        gotPilotOnly = true;
    catch
        try
            H_pilots = nrChannelEstimate(carrier, rxGrid, dmrsIndices, dmrsSymbols, ...
                'CDMLengths', cdmLengths, ...
                'AveragingWindow', [1 1]);
            gotPilotOnly = false;
        catch
            H_pilots = nrChannelEstimate(rxGrid, dmrsIndices, dmrsSymbols);
            gotPilotOnly = false;
        end
    end

    % Our own interpolation (keeps dataset definition stable across releases)
    if gotPilotOnly
        H_est = localInterpPilots2D_fillmissing(H_pilots, pilotSubc, pilotSym, K, L);
    else
        H_est = H_pilots; % already interpolated by toolbox
    end

    % ---- Assertion after interpolation (catches silent failures early) ----
    assert(size(H_est,1)==K && size(H_est,2)==L && size(H_est,3)==cfg.numRxAntennas && size(H_est,4)==pdsch.NumLayers, ...
        'H_est size mismatch after interpolation: got %s, expected [%d %d %d %d]', ...
        mat2str(size(H_est)), K, L, cfg.numRxAntennas, pdsch.NumLayers);

    % Pack to real/imag channels
    Xin = localPackLinksToChannels(H_est);
    Ygt = localPackLinksToChannels(H_perfect);

    % Packed-format sanity check
    expectedC = 2 * cfg.numRxAntennas * cfg.numTxAntennas;
    assert(isreal(Xin) && ndims(Xin)==3 && all(size(Xin)==[K L expectedC]), ...
        "Packed Xin broken at sample %d: size=%s expected=[%d %d %d].", ...
        n, mat2str(size(Xin)), K, L, expectedC);

    assert(isreal(Ygt) && ndims(Ygt)==3 && all(size(Ygt)==[K L expectedC]), ...
        "Packed Ygt broken at sample %d: size=%s expected=[%d %d %d].", ...
        n, mat2str(size(Ygt)), K, L, expectedC);

    % Store tensors (memory or stream)
    if cfg.writeMode == "memory"
        X_input(:,:,:,n) = Xin;
        Y_label(:,:,:,n) = Ygt;
    else
        if useMatStream
            mf.X_input(:,:,:,n) = Xin;
            mf.Y_label(:,:,:,n) = Ygt;
        end
        if useH5Stream
            localH5Write4DSlice(h5Path, "/X_input", Xin, n);
            localH5Write4DSlice(h5Path, "/Y_label", Ygt, n);
        end
    end

    % Quick-QA metrics computed on-the-fly (works in stream mode)
    rxQA = 1; txQA = 1;
    H11_in = H_est(:,:,rxQA,txQA);
    H11_gt = H_perfect(:,:,rxQA,txQA);

    a = double(H11_in(:));
    b = double(H11_gt(:));

    corrH11Log(n) = single(abs((a'*b) / (norm(a)*norm(b) + eps)));
    e = a - b;
    nmseH11Log(n) = single((norm(e)^2) / (norm(b)^2 + eps));

    % Keep one example label for PDP quick-look
    if cfg.doPDPQuickLook && isempty(exampleIdx)
        exampleIdx = n;
        exampleHgt = H_perfect;  % K x L x NRx x NTx (complex)
    end

    % Progress
    if cfg.showProgress && (mod(n, max(1,round(N/10))) == 0)
        fprintf("Progress: %3.0f%%\n", 100*n/N);
    end
end

%% ============================ METADATA ===================================
datasetMeta = struct();
datasetMeta.Name       = cfg.datasetName;
datasetMeta.Version    = cfg.datasetVer;
datasetMeta.Author     = cfg.authorName;
datasetMeta.CreatedOn  = datetime;
datasetMeta.RandomSeed = cfg.randomSeed;

datasetMeta.TensorFormat   = "X_input[K,L,2*NRx*NTx,N], Y_label[K,L,2*NRx*NTx,N]";
datasetMeta.ChannelPacking = "linkIndex=(tx-1)*NRx+rx; Re=2*(linkIndex-1)+1; Im=Re+1";
datasetMeta.NTx = cfg.numTxAntennas;
datasetMeta.NRx = cfg.numRxAntennas;

datasetMeta.NSizeGrid  = carrier.NSizeGrid;
datasetMeta.SCS_kHz    = carrier.SubcarrierSpacing;
datasetMeta.NumSubc    = K;
datasetMeta.NumSym     = L;
datasetMeta.SampleRate = ofdmInfo.SampleRate;
datasetMeta.Nfft       = ofdmInfo.Nfft;

datasetMeta.DelayProfiles       = cellstr(cfg.delayProfiles);
datasetMeta.DelaySpreadRange_ns = cfg.delaySpreadRange_s * 1e9;
datasetMeta.SNRRange_dB         = cfg.snrRange_dB;
datasetMeta.DopplerMode         = cfg.dopplerMode;
datasetMeta.DopplerRange_Hz     = cfg.DopplerRange_Hz;
datasetMeta.CarrierFrequencyHz  = cfg.carrierFrequencyHz;

datasetMeta.InputDefinition = "DM-RS estimate at pilots + 2D interpolation (fillmissing linear)";
datasetMeta.LabelDefinition = "Perfect ofdm-response from nrTDLChannel";

datasetMeta.PilotRE_Unique       = pilotRE_unique;
datasetMeta.PilotDensity_Unique  = pilotDensity_unique;
datasetMeta.PilotRE_PerPort      = pilotRE_perPort;
datasetMeta.PilotDensity_PerPort = pilotDensity_perPort;

datasetMeta.ParamSampling = cfg.paramSampling;
datasetMeta.CDMLengthsUsed = cdmLengths;


%% ===================== QUICK QA (LIGHTWEIGHT) ============================
if cfg.doQuickQA

    fprintf('\n================== QUICK QA SUMMARY ==================\n');
    fprintf('Dataset: %s %s\n', cfg.datasetName, cfg.datasetVer);
    fprintf('Author:  %s\n', cfg.authorName);
    fprintf('BaseName: %s\n', cfg.baseName);
    fprintf('Save folder: <repo>/data\n');

    % ---- print values ----
    fprintf('Dims: K=%d subcarriers, L=%d OFDM symbols, N=%d samples\n', K, L, N);
    fprintf('Antennas: NTx=%d, NRx=%d, numLinks=%d, packedChannels=%d\n', ...
        cfg.numTxAntennas, cfg.numRxAntennas, ...
        cfg.numTxAntennas*cfg.numRxAntennas, ...
        2*cfg.numTxAntennas*cfg.numRxAntennas);

    % --- Pilot density (hardness anchor reviewers like) ---
    fprintf('\n--- Pilot density (TF overhead) ---\n');
    fprintf('Unique DM-RS REs: %d / %d => density = %.4f (%.2f%%)\n', ...
        pilotRE_unique, K*L, pilotDensity_unique, 100*pilotDensity_unique);

    fprintf('Per-port DM-RS REs (count): %s\n', mat2str(pilotRE_perPort(:).'));
    fprintf('Per-port density (%%):       %s\n', mat2str((100*pilotDensity_perPort(:)).', 4));


    localPrintStats = @(name,x,scale,unitFmt) ...
        fprintf('%-10s: mean=%8.2f | std=%8.2f | min=%8.2f | P5=%8.2f | P50=%8.2f | P95=%8.2f | max=%8.2f %s\n', ...
        name, mean(x,'omitnan')*scale, std(x,'omitnan')*scale, min(x,[],'omitnan')*scale, ...
        prctile(x,5)*scale, prctile(x,50)*scale, prctile(x,95)*scale, max(x,[],'omitnan')*scale, unitFmt);

    fprintf('\n--- Scenario coverage (basic stats) ---\n');
    localPrintStats('SNR',    snrLog_dB,        1,    'dB');
    localPrintStats('DS',     delaySpreadLog_s, 1e9,  'ns');
    localPrintStats('Doppler',dopplerLog_Hz,    1,    'Hz');
    localPrintStats('Speed',  speedLog_mps,     1,    'm/s');

    statsRow = @(x,scale) [ ...
        mean(x,'omitnan')*scale, std(x,'omitnan')*scale, min(x,[],'omitnan')*scale, ...
        prctile(x,5)*scale, prctile(x,50)*scale, prctile(x,95)*scale, max(x,[],'omitnan')*scale ];

    S = [ ...
        statsRow(snrLog_dB,        1); ...
        statsRow(delaySpreadLog_s, 1e9); ...
        statsRow(dopplerLog_Hz,    1); ...
        statsRow(speedLog_mps,     1) ];

    ScenarioTable = array2table(S, ...
        'VariableNames', {'Mean','Std','Min','P5','P50','P95','Max'}, ...
        'RowNames', {'SNR_dB','DelaySpread_ns','Doppler_Hz','Speed_mps'});

    disp('--- Scenario coverage (basic stats) ---');
    disp(ScenarioTable);

    [bestCorr,bestIdx]   = max(corrH11Log);
    [worstCorr,worstIdx] = min(corrH11Log);

    fprintf('\n--- Pilot+Interp sanity (H11 only) ---\n');
    fprintf('H11 Corr: mean=%.4f | P5=%.4f | P50=%.4f | P95=%.4f | min=%.4f | max=%.4f\n', ...
        mean(corrH11Log), prctile(corrH11Log,5), prctile(corrH11Log,50), prctile(corrH11Log,95), worstCorr, bestCorr);
    fprintf('H11 NMSE: mean=%.4f | P5=%.4f | P50=%.4f | P95=%.4f\n', ...
        mean(nmseH11Log), prctile(nmseH11Log,5), prctile(nmseH11Log,50), prctile(nmseH11Log,95));

    fprintf('Best idx=%d: SNR=%.1f dB | DS=%.1f ns | fd=%.0f Hz | v=%.2f m/s | Profile=%s\n', ...
        bestIdx, snrLog_dB(bestIdx), delaySpreadLog_s(bestIdx)*1e9, dopplerLog_Hz(bestIdx), speedLog_mps(bestIdx), profileLog(bestIdx));
    fprintf('Worst idx=%d: SNR=%.1f dB | DS=%.1f ns | fd=%.0f Hz | v=%.2f m/s | Profile=%s\n', ...
        worstIdx, snrLog_dB(worstIdx), delaySpreadLog_s(worstIdx)*1e9, dopplerLog_Hz(worstIdx), speedLog_mps(worstIdx), profileLog(worstIdx));

    % --- Hardness rate: hardest-corner percentage (single number) ---
    snrLow = prctile(snrLog_dB,20);
    fdHigh = prctile(dopplerLog_Hz,80);
    dsHigh = prctile(delaySpreadLog_s,80);

    isHard = (snrLog_dB <= snrLow) & (dopplerLog_Hz >= fdHigh) & (delaySpreadLog_s >= dsHigh);
    hardCount = nnz(isHard);
    hardPct   = 100 * hardCount / numel(isHard);

    fprintf('\n--- Hardness rate (hardest corner) ---\n');
    fprintf('Hardest-corner samples: %d / %d (%.2f%%)\n', hardCount, numel(isHard), hardPct);
    fprintf('Definition: SNR<=P20(%.2f dB), Doppler>=P80(%.0f Hz), DS>=P80(%.1f ns)\n', ...
        snrLow, fdHigh, dsHigh*1e9);


    fprintf('======================================================\n\n');
end


%% ============================ PDP QUICK LOOK =============================
%  This "effective PDP" differs from cfg.delaySpread due to:
%   1) Bandwidth limitation (only K active subcarriers)
%   2) FFT windowing effects
%   3) Noise corruption
%   4) Interpolation artifacts
% Use only for sanity-checking delay structure, NOT ground truth validation.
if cfg.doPDPQuickLook && ~isempty(exampleHgt)
    rxPlot = 1; txPlot = 1;
    Hact = exampleHgt(:,:,rxPlot,txPlot);  % K x L (complex)

    [tau_s, pdp] = localEffectivePDP_fftEmbed(Hact, ofdmInfo, K);

    figure('Name','PDP Quick Look (effective)','Position',[120 120 720 360]);
    plot(tau_s*1e9, pdp/(max(pdp)+eps), 'LineWidth', 1.2);
    grid on;
    xlim([0 2200]);
    xlabel('Delay (ns)');
    ylabel('Normalized Power');
    set(gca,'YScale','log'); ylim([1e-5 1]);
    title(sprintf('Effective PDP (FFT-embedded) | sample=%d | rx%d-tx%d', exampleIdx, rxPlot, txPlot));
end

%% ============================ SAVE =======================================
if cfg.saveToDisk

    % MEMORY MODE: write everything at once
    if cfg.saveMat && (cfg.writeMode == "memory")
        fprintf("[INFO] Saving MAT: %s\n", baseNameFinal + ".mat");
        save(matPath, ...
            "X_input","Y_label", ...
            "delaySpreadLog_s","speedLog_mps","dopplerLog_Hz","snrLog_dB", ...
            "seedLog","profileLog","mimoCorrLog", ...
            "corrH11Log","nmseH11Log", ...
            "carrier","pdsch","datasetMeta","cfg", ...
            "-v7.3");
        fprintf("[SAVED] %s\n", baseNameFinal + ".mat");
    end

    % STREAM MODE: append logs/meta into MAT created by matfile
    if cfg.saveMat && (cfg.writeMode == "stream")
        mf.delaySpreadLog_s = delaySpreadLog_s;
        mf.speedLog_mps     = speedLog_mps;
        mf.dopplerLog_Hz    = dopplerLog_Hz;
        mf.snrLog_dB        = snrLog_dB;
        mf.seedLog          = seedLog;
        mf.profileLog       = profileLog;
        mf.mimoCorrLog      = mimoCorrLog;
        mf.corrH11Log       = corrH11Log;
        mf.nmseH11Log       = nmseH11Log;
        mf.carrier          = carrier;
        mf.pdsch            = pdsch;
        mf.datasetMeta      = datasetMeta;
        mf.cfg              = cfg;
        fprintf("[SAVED] %s\n", baseNameFinal + ".mat");
        clear mf; % close handle
    end

    % H5 logs/meta (works for both stream and memory)
    if cfg.saveH5
        if exist(h5Path,"file") || useH5Stream
            % Write logs
            localWriteH5Dataset(h5Path, "/logs/delaySpreadLog_s", delaySpreadLog_s, cfg.h5DeflateLevel);
            localWriteH5Dataset(h5Path, "/logs/speedLog_mps",      speedLog_mps,      cfg.h5DeflateLevel);
            localWriteH5Dataset(h5Path, "/logs/dopplerLog_Hz",     dopplerLog_Hz,     cfg.h5DeflateLevel);
            localWriteH5Dataset(h5Path, "/logs/snrLog_dB",         snrLog_dB,         cfg.h5DeflateLevel);
            localWriteH5Dataset(h5Path, "/logs/seedLog",           seedLog,           cfg.h5DeflateLevel);
            localWriteH5Dataset(h5Path, "/logs/corrH11Log",        corrH11Log,        cfg.h5DeflateLevel);
            localWriteH5Dataset(h5Path, "/logs/nmseH11Log",        nmseH11Log,        cfg.h5DeflateLevel);

            localWriteH5StringVector(h5Path, "/logs/profileLog",  profileLog);
            localWriteH5StringVector(h5Path, "/logs/mimoCorrLog", mimoCorrLog);

            % Meta JSON
            localWriteH5StringScalar(h5Path, "/meta/datasetMeta_json", jsonencode(datasetMeta));
            localWriteH5StringScalar(h5Path, "/meta/cfg_json", jsonencode(cfg));

            % Root attrs (handy for readers)
            h5writeatt(h5Path, "/", "TensorFormat", datasetMeta.TensorFormat);
            h5writeatt(h5Path, "/", "ChannelOrder", datasetMeta.ChannelPacking);
            h5writeatt(h5Path, "/", "NTx", int32(cfg.numTxAntennas));
            h5writeatt(h5Path, "/", "NRx", int32(cfg.numRxAntennas));
            h5writeatt(h5Path, "/", "NumSamples", int32(N));
            h5writeatt(h5Path, "/", "K_Subcarriers", int32(K));
            h5writeatt(h5Path, "/", "L_Symbols", int32(L));

            fprintf("[SAVED] %s\n", baseNameFinal + ".h5");
        else
            warning("H5 path not found; tensors may not have been created. Check save settings.");
        end
    end
end

disp("Done.");

%% ============================ LOCAL FUNCTIONS =============================

function u = localStratified01(N)
% Returns stratified samples in [0,1], one per bin, randomly jittered.
    u = ((0:N-1)' + rand(N,1)) / N;
end

function rxGridOut = localFixRxGridSize(rxGridIn,K,L)
% Ensure rxGrid is K-by-L-by-NRx. Pad/truncate if needed.
% Handles empty rxGridIn safely (edge case).
    % if isempty(rxGridIn)
    %     rxGridOut = zeros(K,L,1,'like',rxGridIn);
    %     return;
    % end
    if isempty(rxGridIn)
        rxGridOut = zeros(K,L,1,'single');
        return;
    end

    rxGridOut = rxGridIn;

    if size(rxGridOut,1) ~= K
        if size(rxGridOut,1) > K
            rxGridOut = rxGridOut(1:K,:,:);
        else
            rxGridOut(end+1:K,:,:) = 0;
        end
    end

    if size(rxGridOut,2) > L
        rxGridOut = rxGridOut(:,1:L,:);
    elseif size(rxGridOut,2) < L
        rxGridOut(:,end+1:L,:) = 0;
    end
end

function Hfull = localInterpPilots2D_fillmissing(Hpilots,pilotSubc,pilotSym,K,L)
% Interpolate pilot-only channel estimates across full KxL grid.
% Uses fillmissing on real/imag separately (robust for complex arrays).

    NRx = size(Hpilots,3);
    P   = size(Hpilots,4);

    % Use real datatype for NaNs (IMPORTANT; avoids complex(nan, nan) errors)
    likeReal = real(Hpilots(1));

    Hfull = complex(zeros(K,L,NRx,P,'like',likeReal), zeros(K,L,NRx,P,'like',likeReal));

    for p = 1:P
        kPil = pilotSubc{p};
        lPil = pilotSym{p};
        linPil = sub2ind([K L], kPil, lPil);

        for r = 1:NRx
            Hp = Hpilots(:,:,r,p);   % KxL (complex), mostly NaN except pilots

            Hre = nan(K,L,'like',likeReal);
            Him = nan(K,L,'like',likeReal);

            HpPil = Hp(linPil);
            Hre(linPil) = real(HpPil);
            Him(linPil) = imag(HpPil);

            % Fill along frequency then time (edges -> nearest)
            Hre = fillmissing(Hre,"linear",1,"EndValues","nearest");
            Hre = fillmissing(Hre,"linear",2,"EndValues","nearest");

            Him = fillmissing(Him,"linear",1,"EndValues","nearest");
            Him = fillmissing(Him,"linear",2,"EndValues","nearest");

            Hfull(:,:,r,p) = complex(Hre,Him);
        end
    end
end

function X = localPackLinksToChannels(H)
% Convert H(K,L,NRx,NTx) complex -> X(K,L,2*NRx*NTx) single.
    [K,L,NRx,NTx] = size(H);
    numLinks = NRx*NTx;

    X = zeros(K,L,2*numLinks,"single");

    ch = 0;
    for tx = 1:NTx
        for rx = 1:NRx
            ch = ch + 1;
            Hrt = H(:,:,rx,tx);
            X(:,:,2*(ch-1)+1) = single(real(Hrt));
            X(:,:,2*(ch-1)+2) = single(imag(Hrt));
        end
    end
end

function localH5Create4D(h5Path, dsetName, sz4, deflateLevel)
% Create a 4-D dataset once (chunked for sample streaming).
    sz4 = double(sz4(:).');                 % force row vector
    assert(numel(sz4)==4, "sz4 must be 1x4: [K L C N]. Got %d elems.", numel(sz4));

    K = sz4(1); L = sz4(2); C = sz4(3);
    chunk = [min(K,64) min(L,14) min(C,16) 1];
    chunk = min(chunk, sz4);
    chunk(chunk < 1) = 1;

    h5create(h5Path, dsetName, sz4, ...
        "Datatype","single", ...
        "ChunkSize",chunk, ...
        "Deflate",deflateLevel);
end

function localH5Write4DSlice(h5Path, dsetName, Xslice, n)
% Write one sample slice into a fixed-size 4-D dataset at index n.
% Xslice is K x L x C (single).
    sz = size(Xslice);
    start = [1 1 1 n];
    count = [sz(1) sz(2) sz(3) 1];
    h5write(h5Path, dsetName, Xslice, start, count);
end

function [tau, pdp] = localEffectivePDP_fftEmbed(Hact, ofdmInfo, K)
% Effective PDP derived from OFDM frequency response using correct FFT length.
% Avoids wrap-around spike you get if you IFFT only over K.
%
% IMPORTANT: This "effective PDP" differs from cfg.delaySpread due to:
%   1) Bandwidth limitation (only K active subcarriers)
%   2) FFT windowing effects
%   3) Noise corruption
%   4) Interpolation artifacts
% Use only for sanity-checking delay structure, NOT ground truth validation.

    Nfft = ofdmInfo.Nfft;
    Fs   = ofdmInfo.SampleRate;
    Lsym = size(Hact,2);

    Hshift = complex(zeros(Nfft, Lsym, "like", Hact));

    halfK = floor(K/2);
    dc = Nfft/2 + 1;

    Hshift(dc-halfK:dc-1,:) = Hact(1:halfK,:);
    Hshift(dc+1:dc+halfK,:) = Hact(halfK+1:2*halfK,:);

    Hifft = ifftshift(Hshift,1);
    h = ifft(Hifft, Nfft, 1);

    pdp = mean(abs(h).^2,2);
    pdp = pdp / (sum(pdp) + eps);

    tau = (0:Nfft-1).' / Fs;

    [~,iPeak] = max(pdp);
    pdp = circshift(pdp, -(iPeak-1));

    halfN = floor(Nfft/2);
    tau = tau(1:halfN);
    pdp = pdp(1:halfN);
    pdp = pdp / (sum(pdp) + eps);
end

function localWriteH5Dataset(h5Path, dsetName, X, deflateLevel)
% Write numeric arrays to H5 with chunking + compression.
    if isempty(X); return; end

    sz = size(X);

    chunk = sz;
    if numel(sz) >= 1; chunk(1) = min(sz(1), 64); end
    if numel(sz) >= 2; chunk(2) = min(sz(2), 14); end
    if numel(sz) >= 3; chunk(3) = min(sz(3), 16); end
    if numel(sz) >= 4; chunk(4) = 1; end

    if isvector(X)
        if sz(1) == 1
            chunk = [1 min(sz(2),4096)];
        else
            chunk = [min(sz(1),4096) 1];
        end
    end

    % Create fresh dataset; if it already exists, delete the file beforehand.
    h5create(h5Path, dsetName, sz, ...
        "Datatype", class(X), ...
        "ChunkSize", chunk, ...
        "Deflate", deflateLevel);

    h5write(h5Path, dsetName, X);
end

function localWriteH5StringVector(h5Path, dsetName, strVec)
% Store string vector as uint8 padded char matrix [N x maxLen].
    if isempty(strVec); strVec = strings(0,1); end
    cellVec = cellstr(string(strVec));
    nRows = numel(cellVec);

    if nRows == 0
        charMat = ' ';
    else
        lens   = cellfun(@numel, cellVec);
        maxLen = max(1, max(lens));
        charMat = repmat(' ', nRows, maxLen);
        for i = 1:nRows
            s = cellVec{i};
            if isempty(s); s = ''; end
            charMat(i, 1:numel(s)) = s;
        end
    end

    h5create(h5Path, dsetName, size(charMat), "Datatype","uint8");
    h5write(h5Path, dsetName, uint8(charMat));
end

function localWriteH5StringScalar(h5Path, dsetName, s)
% Store one JSON/string as uint8 row vector.
    if isstring(s); s = char(s); end
    if isempty(s); s = '{}'; end
    u = uint8(s(:).');
    h5create(h5Path, dsetName, size(u), "Datatype","uint8");
    h5write(h5Path, dsetName, u);
end

