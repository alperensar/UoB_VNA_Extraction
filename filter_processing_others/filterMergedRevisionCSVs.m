function filterMergedRevisionCSVs()
% filterMergedRevisionCSVs (JSON-driven, 2 input types)
%
% INPUT OPTIONS:
%   1) Band-level merged:
%       - Reads Methods and Bands from the JSON config.
%       - Under sample/method/band, it looks for the JSON "MergedSubFolder"
%         (e.g., "Merged_Revision" or "Merged").
%       - Creates a sibling folder "Merged_Filtered_Revision".
%       - Filters merged/averaged CSVs using filterUoBPermittivity (band-based),
%         preserving the existing behavior.
%
%   2) Full-band averaged S2P:
%       - Under sample/method, it looks for "AveragedLast_Revision_S2P".
%       - Filters all CSVs inside (full-band eps/tan curves).
%       - Deletes rows outside the JSON band frequency ranges (removes fitted tails).
%       - Writes results to "AveragedLastProcessed_S2P".
%
% Filtering mode:
%   • Interactive: GUI opens for every curve.
%   • Band auto-apply: For the same fileKey + signal_type, GUI opens once;
%     remaining curves reuse the saved settings silently.
%
% IMPORTANT UPDATE:
%   - JSON now includes a dedicated S2P averaged suffix:
%       config.MergedProcessing.Output.S2PAveragedSuffix = "_S2PAveraged_Data.csv"
%   - Base files matching these suffixes are copied (not filtered):
%       PermittivityMergedSuffix, PermittivityAveragedSuffix,
%       S2PMergedSuffix, S2PAveragedSuffix
%
% EXTRA REQUIREMENTS:
%   - In each folder, roughness and without-roughness datasets must use the SAME filters.
%     (This is enforced by using shared logical keys (fakeKeyEps/fakeKeyTan) per folder/mode.)
%   - All applied filter settings are appended to a TXT file in the MAIN ROOT folder.

    %------------------------------------------------------
    % 0) Which input type should be processed?
    %------------------------------------------------------
    targetChoice = questdlg( ...
        ['Select what to filter:' newline newline ...
         '  • Band-level merged (' ...
         'JSON MergedSubFolder, e.g. Merged_Revision / Merged):' newline ...
         '      - Per band, per file filtering inside the merged folder.' newline ...
         '  • Full-band averaged S2P (AveragedLast_Revision_S2P):' newline ...
         '      - Final stitched S2P curves (all bands together) will be' newline ...
         '        filtered once over full frequency, then cropped to band ranges.' ], ...
        'Input selection', ...
        'Band-level merged (JSON MergedSubFolder)', ...
        'Full-band averaged S2P (AveragedLast_Revision_S2P)', ...
        'Band-level merged (JSON MergedSubFolder)');

    if isempty(targetChoice)
        disp('Operation cancelled (no input type selected).');
        return;
    end

    if startsWith(targetChoice, 'Band-level')
        targetMode = "merged";        % legacy behavior
    else
        targetMode = "last_s2p";      % new behavior
    end

    %------------------------------------------------------
    % 1) Filtering workflow mode (GUI frequency)
    %------------------------------------------------------
    modeChoice = questdlg( ...
        ['Select filtering mode:' newline newline ...
         '  • Interactive: GUI opens for every curve (you can tweak each time).' newline ...
         '  • Band auto-apply: First time for each logical key (fileKey + signal_type)' newline ...
         '    opens the GUI; remaining curves reuse the same settings silently.'], ...
        'Filtering mode', ...
        'Interactive (per curve)', ...
        'Band auto-apply (per key)', ...
        'Interactive (per curve)');

    if isempty(modeChoice)
        disp('Operation cancelled by user (no mode selected).');
        return;
    end

    if startsWith(modeChoice, 'Interactive')
        filterMode = "interactive";
    else
        filterMode = "band_auto";
    end

    %------------------------------------------------------
    % 2) Load JSON config
    %------------------------------------------------------
    configPath = 'config_VNA.json';
    if ~isfile(configPath)
        error('Could not find config file: %s', configPath);
    end
    configText = fileread(configPath);
    config = jsondecode(configText);

    if ~isfield(config, "MergedProcessing")
        error('JSON does not contain "MergedProcessing" section.');
    end

    mp = config.MergedProcessing;
    if ~isfield(mp, "Methods")
        error('JSON MergedProcessing section does not contain "Methods".');
    end

    methodsCfg = mp.Methods;
    outCfg = mp.Output;

    % Settings needed for band-level merged mode (also safe to read in last_s2p mode)
    mergedSubFolderName     = char(outCfg.MergedSubFolder);             % e.g. "Merged_Revision" or "Merged"
    permMergedSuffix        = char(outCfg.PermittivityMergedSuffix);    % e.g. "_Merged_Data.csv"
    permAveragedSuffix      = char(outCfg.PermittivityAveragedSuffix);  % e.g. "_Averaged_Data.csv"
    s2pMergedSuffix         = char(outCfg.S2PMergedSuffix);             % e.g. "_S2PMerged.csv"

    % NEW: dedicated S2P averaged suffix
    s2pAveragedSuffix = '';
    if isfield(outCfg, "S2PAveragedSuffix")
        s2pAveragedSuffix = char(outCfg.S2PAveragedSuffix);             % e.g. "_S2PAveraged_Data.csv"
    end

    %------------------------------------------------------
    % 3) Root folder selection (MAIN ROOT)
    %------------------------------------------------------
    if targetMode == "merged"
        introMsg = [
            "This script will:" + newline + newline + ...
            "  • Use JSON to discover Methods, Bands and merged folder name" + newline + ...
            "  • For each sample/method/band, it will look for '" + mergedSubFolderName + "'" + newline + ...
            "  • For each, it will create 'Merged_Filtered_Revision' and process CSVs inside." + newline + newline + ...
            "SKIP (no filtering, just copy):" + newline + ...
            "  - *" + permMergedSuffix     + newline + ...
            "  - *" + permAveragedSuffix   + newline + ...
            "  - *" + s2pMergedSuffix      + newline + ...
            "  - *" + string(s2pAveragedSuffix) + newline + newline + ...
            "FILTER (filterUoBPermittivity):" + newline + ...
            "  - All other *_Merged*.csv / *_Averaged*.csv files." + newline + newline + ...
            "Mode: " + string(filterMode) ...
        ];
        titleStr = 'Filter band-level merged data';
    else
        introMsg = [
            "This script will:" + newline + newline + ...
            "  • For each sample/method, look for 'AveragedLast_Revision_S2P'." + newline + ...
            "  • Inside, read FINAL stitched S2P averaged CSVs (all bands together)." + newline + ...
            "  • For each eps/tan curve:" + newline + ...
            "        - Run filterUoBPermittivity over FULL frequency range" + newline + ...
            "        - Then delete samples that lie OUTSIDE JSON band frequency ranges" + newline + ...
            "          (i.e. fitted/extrapolated tail beyond bands is removed)." + newline + ...
            "  • Save results to 'AveragedLastProcessed_S2P' (CSV + MAT)." + newline + newline + ...
            "Mode: " + string(filterMode)
        ];
        titleStr = 'Filter full-band AveragedLast_Revision_S2P';
    end

    userChoice = questdlg(introMsg, titleStr, ...
                          'Continue', 'Cancel', 'Continue');
    if ~strcmp(userChoice, 'Continue')
        disp('Operation cancelled by user.');
        return;
    end

    rootDir = uigetdir([], 'Select your MAIN root folder (e.g. RAWDATA)');
    if rootDir == 0
        disp('Operation cancelled by user (no root folder).');
        return;
    end

    %------------------------------------------------------
    % 4) Sample loop
    %------------------------------------------------------
    sampleFolders = dir(rootDir);
    sampleFolders = sampleFolders([sampleFolders.isdir] & ~startsWith({sampleFolders.name}, '.'));

    for s = 1:numel(sampleFolders)
        sampleName      = sampleFolders(s).name;
        sampleFolderDir = fullfile(rootDir, sampleName);

        % For each method (from JSON)
        for m = 1:numel(methodsCfg)
            methodCfg        = methodsCfg(m);
            methodFolderName = char(methodCfg.FolderName);   % "Keysight_PNA_NPL", "Keysight_PNA_UoB", ...
            methodFolderPath = fullfile(sampleFolderDir, methodFolderName);

            if ~exist(methodFolderPath, 'dir')
                continue;
            end

            if targetMode == "merged"
                %=============================================
                % LEGACY: band-level merged folder filtering
                %=============================================
                processBandLevelMerged( ...
                    sampleName, ...
                    methodFolderName, ...
                    methodFolderPath, ...
                    methodCfg, ...
                    mergedSubFolderName, ...
                    permMergedSuffix, ...
                    permAveragedSuffix, ...
                    s2pMergedSuffix, ...
                    s2pAveragedSuffix, ...
                    rootDir, ...
                    filterMode);
            else
                %=============================================
                % NEW: full-band AveragedLast_Revision_S2P
                %=============================================
                processAveragedLastS2PForMethod( ...
                    sampleName, ...
                    methodFolderName, ...
                    methodFolderPath, ...
                    methodCfg, ...
                    rootDir, ...
                    filterMode);
            end
        end
    end

    if targetMode == "merged"
        fprintf('\n🎯 All merged folders processed (filtered copies in Merged_Filtered_Revision).\n');
    else
        fprintf('\n🎯 All AveragedLast_Revision_S2P folders processed (results in AveragedLastProcessed_S2P).\n');
    end
end


%======================================================================%
% Full-band AveragedLast_Revision_S2P processing                        %
%======================================================================%
function processAveragedLastS2PForMethod( ...
            sampleName, ...
            methodFolderName, ...
            methodFolderPath, ...
            methodCfg, ...
            rootDir, ...
            filterMode)

    inDir  = fullfile(methodFolderPath, 'AveragedLast_Revision_S2P');
    if ~exist(inDir, 'dir')
        return;  % No AveragedLast_Revision_S2P for this method
    end

    outDir = fullfile(methodFolderPath, 'AveragedLastProcessed_S2P');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    fprintf('\n=== Sample: %s | Method: %s | AveragedLast_Revision_S2P ===\n', ...
            sampleName, methodFolderName);
    fprintf('    Input folder:  %s\n', inDir);
    fprintf('    Output folder: %s\n', outDir);

    % Read band frequency ranges from JSON as [fmin,fmax] in GHz
    bandRangesGHz = getBandFreqRangesGHz(methodCfg);

    % Use a single shared logical key per method (forces SAME filter for roughness/without-roughness)
    fakeKeyEps = fullfile(outDir, 'FILTER_KEY_LAST_EPS.csv');
    fakeKeyTan = fullfile(outDir, 'FILTER_KEY_LAST_TAN.csv');

    % All CSV files in this folder
    csvFiles = dir(fullfile(inDir, '*.csv'));
    if isempty(csvFiles)
        fprintf('    No CSV files in AveragedLast_Revision_S2P.\n');
        return;
    end

    for k = 1:numel(csvFiles)
        inName = csvFiles(k).name;
        inPath = fullfile(csvFiles(k).folder, inName);
        outCSV = fullfile(outDir, inName);   % same name, different folder
        [~, baseName, ~] = fileparts(inName);
        outMAT = fullfile(outDir, [baseName '.mat']);

        fprintf('  Processing full-band S2P: %s\n', inName);

        try
            T = readtable(inPath, 'VariableNamingRule','preserve');

            % Filter full-band + crop outside band ranges
            [Tfiltered, combinedData, segmentStarts, segmentEnds] = ...
                filterS2PAveragedFullBand(T, rootDir, fakeKeyEps, fakeKeyTan, ...
                                          filterMode, bandRangesGHz);

            % Write CSV
            writetable(Tfiltered, outCSV);

            % Write MAT (combinedData + segment info)
            save(outMAT, 'combinedData', 'segmentStarts', 'segmentEnds');

        catch ME
            fprintf('    ! Error processing %s: %s\n', inName, ME.message);
            % If error, at least copy original
            if ~isfile(outCSV)
                copyfile(inPath, outCSV);
            end
        end
    end
end

%======================================================================%
% Helper: Read band frequency ranges (GHz) from methodCfg.Bands         %
%  - Expected fields: FreqMinGHz/FreqMaxGHz or Fmin_GHz/Fmax_GHz        %
%======================================================================%
function bandRangesGHz = getBandFreqRangesGHz(methodCfg)
    bandRangesGHz = [];

    if ~isfield(methodCfg, 'Bands')
        return;
    end

    bands = methodCfg.Bands;
    for b = 1:numel(bands)
        band = bands(b);

        fmin = [];
        fmax = [];

        if isfield(band, 'FreqMinGHz') && isfield(band, 'FreqMaxGHz')
            fmin = double(band.FreqMinGHz);
            fmax = double(band.FreqMaxGHz);
        elseif isfield(band, 'Fmin_GHz') && isfield(band, 'Fmax_GHz')
            fmin = double(band.Fmin_GHz);
            fmax = double(band.Fmax_GHz);
        end

        if ~isempty(fmin) && ~isempty(fmax) && fmax > fmin
            bandRangesGHz(end+1,:) = [fmin, fmax]; %#ok<AGROW>
        end
    end
end

%======================================================================%
% Filter full-band S2P averaged table + crop outside band ranges        %
%======================================================================%
function [Tout, combinedData, segmentStarts, segmentEnds] = ...
    filterS2PAveragedFullBand(T, rootDir, fakeKeyEps, fakeKeyTan, ...
                              filterMode, bandRangesGHz)

    varNames      = T.Properties.VariableNames;
    varNamesLower = lower(varNames);

    idxFreq = find(contains(varNamesLower, 'freq'), 1);
    if isempty(idxFreq)
        warning('    ! s2p_fullband: no freq column detected. Leaving unchanged.');
        Tout          = T;
        combinedData  = T{:,:};
        segmentStarts = 1;
        segmentEnds   = size(combinedData,1);
        return;
    end

    idxEps = find(contains(varNamesLower, {'eps','epsilon'}) & ...
                  ~contains(varNamesLower, 'std'), 1);
    idxTan = find(contains(varNamesLower, 'tan') & ...
                  ~contains(varNamesLower, 'std'), 1);

    if isempty(idxEps) || isempty(idxTan)
        warning('    ! s2p_fullband: epsilon/tan mean columns not detected. Leaving unchanged.');
        Tout          = T;
        combinedData  = T{:,:};
        segmentStarts = 1;
        segmentEnds   = size(combinedData,1);
        return;
    end

    % Original freq and signals
    freqHz    = T{:, idxFreq};
    freqGHz   = freqHz ./ 1e9;
    epsSignal = T{:, idxEps};
    tanSignal = T{:, idxTan};

    % --- 1) Full-band filtering (GUI + log) ---
    yEps = filterUoBPermittivity(freqGHz, epsSignal, fakeKeyEps, rootDir, filterMode);
    yTan = filterUoBPermittivity(freqGHz, tanSignal, fakeKeyTan, rootDir, filterMode);

    T{:, idxEps} = yEps(:);
    T{:, idxTan} = yTan(:);

    % --- 2) Remove samples outside band frequency ranges ---
    if ~isempty(bandRangesGHz)
        keepMask = false(size(freqGHz));
        for r = 1:size(bandRangesGHz,1)
            fmin = bandRangesGHz(r,1);
            fmax = bandRangesGHz(r,2);
            keepMask = keepMask | (freqGHz >= fmin & freqGHz <= fmax);
        end

        if any(~keepMask)
            T       = T(keepMask, :);
            freqHz  = freqHz(keepMask);
        end
    end

    % --- 3) combinedData + segmentStarts/segmentEnds ---
    combinedData = T{:,:};
    totalRows    = size(combinedData,1);

    if totalRows <= 1
        segmentStarts = 1;
        segmentEnds   = totalRows;
    else
        freqDiff = diff(freqHz);
        gapPoints = find(freqDiff > 2e9);   % same legacy gap criterion

        segmentStarts = [1; gapPoints + 1];
        segmentEnds   = [gapPoints; totalRows];
    end

    Tout = T;
end


%======================================================================%
% Averaged permittivity: freq + epsilon_mean + tan_mean (+ std, thk)    %
%======================================================================%
function Tout = filterPermAveragedTable(T, rootDir, fakeKeyEps, fakeKeyTan, filterMode)
    varNames      = T.Properties.VariableNames;
    varNamesLower = lower(varNames);

    idxFreq = find(contains(varNamesLower, 'freq'), 1);
    if isempty(idxFreq)
        warning('    ! perm_averaged: frequency column not detected. Leaving unchanged.');
        Tout = T;
        return;
    end

    idxEps = find(contains(varNamesLower, {'eps','epsilon'}) & ...
                  ~contains(varNamesLower, 'std'), 1);

    idxTan = find(contains(varNamesLower, 'tan') & ...
                  ~contains(varNamesLower, 'std'), 1);

    if isempty(idxEps) || isempty(idxTan)
        warning('    ! perm_averaged: could not detect epsilon/tan mean columns. Leaving unchanged.');
        Tout = T;
        return;
    end

    freqHz    = T{:, idxFreq};
    freqGHz   = freqHz ./ 1e9;
    epsSignal = T{:, idxEps};
    tanSignal = T{:, idxTan};

    yEps = filterUoBPermittivity(freqGHz, epsSignal, fakeKeyEps, rootDir, filterMode);
    yTan = filterUoBPermittivity(freqGHz, tanSignal, fakeKeyTan, rootDir, filterMode);

    T{:, idxEps} = yEps(:);
    T{:, idxTan} = yTan(:);

    Tout = T;
end

%======================================================================%
% 3-col block: [Freq, Eps, Tan] × N  (S2P merged)                       %
%======================================================================%
function dataOut = filterS2PMergedBlocks(data, rootDir, fakeKeyEps, fakeKeyTan, filterMode)
    [~, nCols] = size(data);
    if mod(nCols, 3) ~= 0
        warning('    ! s2p_merged: number of columns is not multiple of 3. Leaving unchanged.');
        dataOut = data;
        return;
    end

    nBlocks = nCols / 3;
    freqHz  = data(:,1);
    freqGHz = freqHz ./ 1e9;

    dataOut = data;

    for b = 1:nBlocks
        baseC  = (b-1)*3;
        epsCol = baseC + 2;
        tanCol = baseC + 3;

        epsSignal = data(:, epsCol);
        tanSignal = data(:, tanCol);

        yEps = filterUoBPermittivity(freqGHz, epsSignal, fakeKeyEps, rootDir, filterMode);
        yTan = filterUoBPermittivity(freqGHz, tanSignal, fakeKeyTan, rootDir, filterMode);

        dataOut(:, epsCol) = yEps(:);
        dataOut(:, tanCol) = yTan(:);
    end
end

%======================================================================%
% Averaged S2P: freq + epsilon_mean + tan_mean (+ std, thk)             %
%======================================================================%
function Tout = filterS2PAveragedTable(T, rootDir, fakeKeyEps, fakeKeyTan, filterMode)
    varNames      = T.Properties.VariableNames;
    varNamesLower = lower(varNames);

    idxFreq = find(contains(varNamesLower, 'freq'), 1);
    if isempty(idxFreq)
        warning('    ! s2p_averaged: frequency column not detected. Leaving unchanged.');
        Tout = T;
        return;
    end

    idxEps = find(contains(varNamesLower, {'eps','epsilon'}) & ...
                  ~contains(varNamesLower, 'std'), 1);
    idxTan = find(contains(varNamesLower, 'tan') & ...
                  ~contains(varNamesLower, 'std'), 1);

    if isempty(idxEps) || isempty(idxTan)
        warning('    ! s2p_averaged: could not detect epsilon/tan mean columns. Leaving unchanged.');
        Tout = T;
        return;
    end

    freqHz    = T{:, idxFreq};
    freqGHz   = freqHz ./ 1e9;
    epsSignal = T{:, idxEps};
    tanSignal = T{:, idxTan};

    yEps = filterUoBPermittivity(freqGHz, epsSignal, fakeKeyEps, rootDir, filterMode);
    yTan = filterUoBPermittivity(freqGHz, tanSignal, fakeKeyTan, rootDir, filterMode);

    T{:, idxEps} = yEps(:);
    T{:, idxTan} = yTan(:);

    Tout = T;
end

%======================================================================%
% Core filtering function with GUI + band auto-apply logic              %
%======================================================================%
function yOut = filterUoBPermittivity(freqRange, signal, fileKey, rootDir, filterMode)
% yOut = filterUoBPermittivity(freqRange, signal, fileKey, rootDir, filterMode)
%
% freqRange : frequency vector (GHz, used only as x-axis)
% signal    : epsilon_real or tan_delta vector
% fileKey   : logical key written in filter_log.csv
%            (for bands we pass fake paths like ".../FILTER_KEY_EPS.csv")
% rootDir   : main root directory where filter_log.csv and TXT summary are stored
% filterMode: "interactive" or "band_auto"
%
% Behavior:
%   - Detects signal type (epsilon vs tan delta).
%   - Reads / updates filter_log.csv:
%        filePath, period, padding, filter, signal_type,
%        padding_head, padding_tail, fit_frac
%   - Mode "interactive":
%        GUI opens on every call.
%   - Mode "band_auto":
%        First time per key+signalType (per MATLAB run) GUI opens,
%        then remaining curves use the saved settings (no GUI).
%
% Extra:
%   - Writes all applied settings once per sessionKey into:
%       <rootDir>/filter_settings_applied.txt

    %--------------------------------------------------------------
    % 0) Signal type: epsilon vs tan delta
    %--------------------------------------------------------------
    signalType = "epsilon";
    if max(abs(signal)) < 0.5
        signalType = "tandelta";
    end

    %--------------------------------------------------------------
    % 1) Per-run GUI tracking + per-run TXT tracking
    %--------------------------------------------------------------
    persistent shownThisRun writtenTxtThisRun
    if isempty(shownThisRun)
        shownThisRun = containers.Map('KeyType','char','ValueType','logical');
    end
    if isempty(writtenTxtThisRun)
        writtenTxtThisRun = containers.Map('KeyType','char','ValueType','logical');
    end

    sessionKey = sprintf('%s|%s', char(fileKey), char(signalType));

    if isKey(shownThisRun, sessionKey)
        hasShownThisRun = shownThisRun(sessionKey);
    else
        hasShownThisRun = false;
    end

    %--------------------------------------------------------------
    % 2) Ensure vector shapes
    %--------------------------------------------------------------
    signal    = signal(:)';        % 1×N
    freqRange = freqRange(:)';     % 1×N

    if numel(freqRange) ~= numel(signal)
        error('freqRange and signal must have the same length.');
    end

    L  = length(signal);
    Fs = 1;  % arbitrary sampling rate in index domain

    %--------------------------------------------------------------
    % 3) Period candidates from FFT (for initial guess)
    %--------------------------------------------------------------
    Y1 = abs(fft(signal - mean(signal)));
    f1 = Fs * (0:(L/2)) / L;
    Y1 = Y1(1:floor(L/2)+1);
    [~, idx1] = max(Y1(2:end));
    dominant_freq1 = f1(idx1 + 1);

    periods = [];
    if dominant_freq1 > 0
        periods(1) = round(1 / dominant_freq1);
    else
        periods(1) = max(3, floor(L/5)); % fallback
    end

    cropLength = max(3, floor(0.8 * L));
    yCropped   = signal(1:cropLength);
    Y2 = abs(fft(yCropped - mean(yCropped)));
    f2 = Fs * (0:(L/2)) / L;
    Y2 = Y2(1:floor(L/2)+1);
    [~, idx2] = max(Y2(2:end));
    dominant_freq2 = f2(idx2 + 1);

    if dominant_freq2 > 0
        periods(2) = round(1 / dominant_freq2);
    else
        periods(2) = periods(1);
    end

    if periods(2) == numel(signal)
        periods(3) = 300;
    end

    %--------------------------------------------------------------
    % 4) Read log file, if any
    %--------------------------------------------------------------
    logFile   = fullfile(rootDir, 'filter_log.csv');
    hasLog    = false;
    hasEntry  = false;
    periodLog = [];
    headLog   = "mirror";
    tailLog   = "mirror";
    filterTagLog = "SG";    % SG / MOVMEAN / GAUSSIAN
    paddingLog   = true;    % logical
    fitFracLog   = 1.0;     % 0–1

    if exist(logFile, 'file')
        hasLog = true;
        logTable = readtable(logFile, 'Delimiter', ',', ...
                             'TextType', 'string', ...
                             'PreserveVariableNames', true);

        % Backwards compatibility: add fit_frac column if missing
        if ~ismember('fit_frac', logTable.Properties.VariableNames)
            logTable.fit_frac = ones(height(logTable), 1);
        end

        matchIdx = (logTable.filePath == string(fileKey)) & ...
                   (logTable.signal_type == signalType);

        if any(matchIdx)
            hasEntry       = true;
            firstIdx       = find(matchIdx, 1);
            periodLog      = logTable.period(firstIdx);
            headLog        = string(logTable.padding_head(firstIdx));
            tailLog        = string(logTable.padding_tail(firstIdx));
            filterTagLog   = string(logTable.filter(firstIdx));
            paddingStr     = string(logTable.padding(firstIdx));
            paddingLog     = strcmpi(paddingStr, "true") | strcmp(paddingStr, "1");
            fitFracVal     = logTable.fit_frac(firstIdx);

            if ~isnan(fitFracVal) && fitFracVal > 0 && fitFracVal <= 1
                fitFracLog = double(fitFracVal);
            end
        end
    end

    %--------------------------------------------------------------
    % 5) AUTO PATH: band_auto mode + we already showed GUI this run
    %--------------------------------------------------------------
    if filterMode == "band_auto" && hasEntry && hasShownThisRun
        if ~isempty(periodLog)
            period = double(periodLog);
        else
            period = periods(1);
        end
        period = max(3, min(period, L-2));

        paddingHeadMode = char(headLog);
        paddingTailMode = char(tailLog);
        if isempty(paddingHeadMode), paddingHeadMode = 'mirror'; end
        if isempty(paddingTailMode), paddingTailMode = 'mirror'; end

        fitFracTail = fitFracLog;
        if ~(fitFracTail > 0 && fitFracTail <= 1)
            fitFracTail = 1.0;
        end

        curves = computeFilteredCurves(signal, freqRange, period, ...
                                       paddingHeadMode, paddingTailMode, fitFracTail);

        tagUpper = upper(strtrim(filterTagLog));
        if strcmp(tagUpper, "SG")
            if paddingLog
                yOut = curves.ySGpad;
            else
                yOut = curves.ySGno;
            end
        elseif strcmp(tagUpper, "MOVMEAN")
            yOut = curves.yRApad;
        elseif strcmp(tagUpper, "GAUSSIAN")
            yOut = curves.yGaussPad;
        else
            yOut = curves.ySGpad;
        end

        % Write TXT once per sessionKey per run
        if ~isKey(writtenTxtThisRun, sessionKey) || ~writtenTxtThisRun(sessionKey)
            appendFilterSettingTxt(rootDir, fileKey, signalType, period, paddingLog, string(filterTagLog), ...
                                   string(paddingHeadMode), string(paddingTailMode), fitFracTail, "auto");
            writtenTxtThisRun(sessionKey) = true;
        end

        yOut = yOut(:);
        return;
    end

    %--------------------------------------------------------------
    % 6) INTERACTIVE PATH (GUI)
    %--------------------------------------------------------------
    if ~isempty(periodLog)
        period = double(periodLog);
    else
        period = periods(1);
    end
    period = max(3, min(period, L-2));

    paddingHeadMode = char(headLog);
    paddingTailMode = char(tailLog);
    if isempty(paddingHeadMode), paddingHeadMode = 'mirror'; end
    if isempty(paddingTailMode), paddingTailMode = 'mirror'; end

    filterIdxInit = 1;
    tagUpper = upper(strtrim(filterTagLog));
    if strcmp(tagUpper, "SG") && ~paddingLog
        filterIdxInit = 2;
    elseif strcmp(tagUpper, "MOVMEAN")
        filterIdxInit = 3;
    elseif strcmp(tagUpper, "GAUSSIAN")
        filterIdxInit = 4;
    end

    fitFracTail = fitFracLog;
    if ~(fitFracTail > 0 && fitFracTail <= 1)
        fitFracTail = 1.0;
    end

    yOut          = signal(:);
    applyPadding  = true;
    filterTag     = "SG";
    accepted      = false;

    curves = struct('ySGpad',[],'yRApad',[],'yGaussPad',[], ...
                    'ySGno',[],'yRAno',[],'yGaussNo',[]);

    %--------------------------------------------------------------
    % 7) Build GUI
    %--------------------------------------------------------------
    scr = get(0, 'ScreenSize');  % [left bottom width height]
    screen_width  = scr(3);
    screen_height = scr(4);

    figHeight = round(screen_height * 0.7);
    figWidth  = min(round(figHeight * 3), screen_width - 40);
    left      = round((screen_width - figWidth) / 2);
    bottom    = round((screen_height - figHeight) / 2);

    hFig = figure('Color','k', ...
                  'Name','Permittivity filter comparison', ...
                  'NumberTitle','off', ...
                  'Position',[left, bottom, figWidth, figHeight], ...
                  'WindowKeyPressFcn', @escKey);

    % 3 axes
    hAx1 = subplot(1,3,1,'Parent',hFig);
    hAx2 = subplot(1,3,2,'Parent',hFig);
    hAx3 = subplot(1,3,3,'Parent',hFig);

    % --- Bottom controls ---

    uicontrol(hFig,'Style','text', ...
        'Units','normalized', ...
        'Position',[0.01 0.01 0.08 0.04], ...
        'String','Period:', ...
        'ForegroundColor','w', ...
        'BackgroundColor','k', ...
        'HorizontalAlignment','left');

    hPeriodEdit = uicontrol(hFig,'Style','edit', ...
        'Units','normalized', ...
        'Position',[0.09 0.01 0.08 0.045], ...
        'String',num2str(period), ...
        'BackgroundColor',[0.2 0.2 0.2], ...
        'ForegroundColor','w');

    uicontrol(hFig,'Style','text', ...
        'Units','normalized', ...
        'Position',[0.50 0.055 0.10 0.035], ...
        'String','Use first %:', ...
        'ForegroundColor','w', ...
        'BackgroundColor','k', ...
        'HorizontalAlignment','left');

    hFitFracEdit = uicontrol(hFig,'Style','edit', ...
        'Units','normalized', ...
        'Position',[0.60 0.055 0.08 0.035], ...
        'String', num2str(fitFracTail*100), ...
        'BackgroundColor',[0.2 0.2 0.2], ...
        'ForegroundColor','w', ...
        'Callback', @fitFracCallback);

    uicontrol(hFig,'Style','text', ...
        'Units','normalized', ...
        'Position',[0.18 0.01 0.22 0.04], ...
        'String',sprintf('Candidates: %s', sprintf('%d ',periods)), ...
        'ForegroundColor',[0.8 0.8 0.8], ...
        'BackgroundColor','k', ...
        'HorizontalAlignment','left');

    uicontrol(hFig,'Style','pushbutton', ...
        'Units','normalized', ...
        'Position',[0.41 0.01 0.08 0.05], ...
        'String','Update', ...
        'Callback',@updateCallback, ...
        'BackgroundColor',[0.3 0.3 0.3], ...
        'ForegroundColor','w');

    uicontrol(hFig,'Style','text', ...
        'Units','normalized', ...
        'Position',[0.50 0.01 0.07 0.04], ...
        'String','Head:', ...
        'ForegroundColor','w', ...
        'BackgroundColor','k', ...
        'HorizontalAlignment','left');

    hHeadPopup = uicontrol(hFig,'Style','popupmenu', ...
        'Units','normalized', ...
        'Position',[0.57 0.01 0.1 0.045], ...
        'String',{'mirror','sine'}, ...
        'BackgroundColor',[0.2 0.2 0.2], ...
        'ForegroundColor','w', ...
        'Callback',@headPopupCallback);

    uicontrol(hFig,'Style','text', ...
        'Units','normalized', ...
        'Position',[0.68 0.01 0.07 0.04], ...
        'String','Tail:', ...
        'ForegroundColor','w', ...
        'BackgroundColor','k', ...
        'HorizontalAlignment','left');

    hTailPopup = uicontrol(hFig,'Style','popupmenu', ...
        'Units','normalized', ...
        'Position',[0.75 0.01 0.1 0.045], ...
        'String',{'mirror','sine'}, ...
        'BackgroundColor',[0.2 0.2 0.2], ...
        'ForegroundColor','w', ...
        'Callback',@tailPopupCallback);

    if strcmpi(paddingHeadMode,'mirror')
        set(hHeadPopup,'Value',1);
    else
        set(hHeadPopup,'Value',2);
    end
    if strcmpi(paddingTailMode,'mirror')
        set(hTailPopup,'Value',1);
    else
        set(hTailPopup,'Value',2);
    end

    hFilterGroup = uibuttongroup('Parent',hFig, ...
        'Units','normalized', ...
        'Position',[0.86 0.01 0.13 0.2], ...
        'Title','Filter to apply', ...
        'ForegroundColor','w', ...
        'BackgroundColor','k');

    hRb1 = uicontrol(hFilterGroup,'Style','radiobutton', ...
        'Units','normalized', ...
        'Position',[0.05 0.75 0.9 0.2], ...
        'String','SG (pad)', ...
        'Tag','SG_PAD', ...
        'BackgroundColor','k', ...
        'ForegroundColor','w');

    hRb2 = uicontrol(hFilterGroup,'Style','radiobutton', ...
        'Units','normalized', ...
        'Position',[0.05 0.50 0.9 0.2], ...
        'String','SG (no pad)', ...
        'Tag','SG_NOPAD', ...
        'BackgroundColor','k', ...
        'ForegroundColor','w');

    hRb3 = uicontrol(hFilterGroup,'Style','radiobutton', ...
        'Units','normalized', ...
        'Position',[0.05 0.25 0.9 0.2], ...
        'String','SG+MovMean', ...
        'Tag','MOVMEAN', ...
        'BackgroundColor','k', ...
        'ForegroundColor','w');

    hRb4 = uicontrol(hFilterGroup,'Style','radiobutton', ...
        'Units','normalized', ...
        'Position',[0.05 0.00 0.9 0.2], ...
        'String','SG+Gaussian', ...
        'Tag','GAUSSIAN', ...
        'BackgroundColor','k', ...
        'ForegroundColor','w');

    switch filterIdxInit
        case 1, set(hFilterGroup,'SelectedObject',hRb1);
        case 2, set(hFilterGroup,'SelectedObject',hRb2);
        case 3, set(hFilterGroup,'SelectedObject',hRb3);
        case 4, set(hFilterGroup,'SelectedObject',hRb4);
    end

    uicontrol(hFig,'Style','pushbutton', ...
        'Units','normalized', ...
        'Position',[0.90 0.88 0.08 0.06], ...
        'String','Accept', ...
        'BackgroundColor',[0.1 0.5 0.1], ...
        'ForegroundColor','w', ...
        'Callback',@acceptCallback);

    uicontrol(hFig,'Style','pushbutton', ...
        'Units','normalized', ...
        'Position',[0.90 0.80 0.08 0.06], ...
        'String','Cancel', ...
        'BackgroundColor',[0.6 0.1 0.1], ...
        'ForegroundColor','w', ...
        'Callback',@cancelCallback);

    recomputeAndPlot();

    % Mark that GUI has been shown for this sessionKey
    shownThisRun(sessionKey) = true;

    uiwait(hFig);

    if isvalid(hFig)
        close(hFig);
    end

    if ~accepted
        yOut = signal(:);
        return;
    end

    %--------------------------------------------------------------
    % 8) Write / update filter_log.csv
    %--------------------------------------------------------------
    newEntry = {
        string(fileKey), ...
        period, ...
        string(applyPadding), ...
        filterTag, ...
        signalType, ...
        string(paddingHeadMode), ...
        string(paddingTailMode), ...
        fitFracTail ...
    };

    if hasLog
        if ~ismember('fit_frac', logTable.Properties.VariableNames)
            logTable.fit_frac = ones(height(logTable), 1);
        end

        matchIdx = (logTable.filePath == string(fileKey)) & ...
                   (logTable.signal_type == signalType);

        if any(matchIdx)
            logTable.period(matchIdx)       = period;
            logTable.padding(matchIdx)      = string(applyPadding);
            logTable.filter(matchIdx)       = filterTag;
            logTable.signal_type(matchIdx)  = signalType;
            logTable.padding_head(matchIdx) = string(paddingHeadMode);
            logTable.padding_tail(matchIdx) = string(paddingTailMode);
            logTable.fit_frac(matchIdx)     = fitFracTail;
        else
            logTable = [logTable; cell2table(newEntry, ...
                'VariableNames', {'filePath','period','padding','filter', ...
                                  'signal_type','padding_head','padding_tail','fit_frac'})];
        end

        writetable(logTable, logFile);
    else
        header   = {'filePath','period','padding','filter', ...
                    'signal_type','padding_head','padding_tail','fit_frac'};
        logTable = cell2table(newEntry, 'VariableNames', header);
        writetable(logTable, logFile);
    end

    % Write TXT once per sessionKey per run
    if ~isKey(writtenTxtThisRun, sessionKey) || ~writtenTxtThisRun(sessionKey)
        appendFilterSettingTxt(rootDir, fileKey, signalType, period, applyPadding, filterTag, ...
                               string(paddingHeadMode), string(paddingTailMode), fitFracTail, "interactive");
        writtenTxtThisRun(sessionKey) = true;
    end

    yOut = yOut(:);

    %==================== NESTED CALLBACKS ====================%
    function escKey(~,event)
        if strcmp(event.Key,'escape')
            uiresume(hFig);
        end
    end

    function headPopupCallback(~,~)
        val   = get(hHeadPopup,'Value');
        items = get(hHeadPopup,'String');
        paddingHeadMode = items{val};
        recomputeAndPlot();
    end

    function tailPopupCallback(~,~)
        val   = get(hTailPopup,'Value');
        items = get(hTailPopup,'String');
        paddingTailMode = items{val};
        recomputeAndPlot();
    end

    function fitFracCallback(~, ~)
        valStr = get(hFitFracEdit, 'String');
        valNum = str2double(valStr);

        if isnan(valNum) || valNum <= 0 || valNum > 100
            warndlg('Fit percentage must be in (0, 100].','Invalid fit %');
            set(hFitFracEdit, 'String', num2str(fitFracTail * 100));
            return;
        end

        fitFracTail = max(0.1, min(valNum/100, 1.0));
        set(hFitFracEdit, 'String', num2str(fitFracTail*100));
        recomputeAndPlot();
    end

    function updateCallback(~,~)
        p = str2double(get(hPeriodEdit,'String'));
        if isnan(p) || p < 2
            warndlg('Period must be >= 2','Invalid period');
            return;
        end
        period = round(p);
        set(hPeriodEdit,'String',num2str(period));
        recomputeAndPlot();
    end

    function recomputeAndPlot()
        curves = computeFilteredCurves(signal, freqRange, period, ...
                                       paddingHeadMode, paddingTailMode, fitFracTail);

        axes(hAx1); cla(hAx1);
        plot(hAx1, freqRange, signal,'w--','LineWidth',1); hold(hAx1,'on');
        plot(hAx1, freqRange, curves.ySGpad,'g','LineWidth',1.2);
        plot(hAx1, freqRange, curves.yRApad,'r','LineWidth',1.2);
        plot(hAx1, freqRange, curves.yGaussPad,'b','LineWidth',1.2);
        grid(hAx1,'on');
        title(hAx1, sprintf('With padding (period %d)',period),'Color','w');
        legend(hAx1,{'Original','SG','SG+MovMean','SG+Gaussian'}, ...
               'Location','best','TextColor','w');
        set(hAx1,'Color','k','XColor','w','YColor','w');

        axes(hAx2); cla(hAx2);
        plot(hAx2, freqRange, signal,'w--','LineWidth',1); hold(hAx2,'on');
        plot(hAx2, freqRange, curves.ySGno,'g','LineWidth',1.2);
        plot(hAx2, freqRange, curves.yRAno,'r','LineWidth',1.2);
        plot(hAx2, freqRange, curves.yGaussNo,'b','LineWidth',1.2);
        grid(hAx2,'on');
        title(hAx2,'Without padding','Color','w');
        legend(hAx2,{'Original','SG','SG+MovMean','SG+Gaussian'}, ...
               'Location','best','TextColor','w');
        set(hAx2,'Color','k','XColor','w','YColor','w');

        axes(hAx3); cla(hAx3);
        plot(hAx3, signal,'w--','LineWidth',1); hold(hAx3,'on');
        plot(hAx3, curves.ySGpad,'r','LineWidth',1.5);
        plot(hAx3, curves.ySGno,'g','LineWidth',1.5);
        grid(hAx3,'on');
        title(hAx3,'SG padded (red) vs SG no-pad (green)','Color','w');
        legend(hAx3,{'Original','SG padded','SG no pad'}, ...
               'Location','best','TextColor','w');
        set(hAx3,'Color','k','XColor','w','YColor','w');
    end

    function acceptCallback(~,~)
        selObj = get(hFilterGroup,'SelectedObject');
        tag    = get(selObj,'Tag');

        switch tag
            case 'SG_PAD'
                yOut         = curves.ySGpad;
                applyPadding = true;
                filterTag    = "SG";
            case 'SG_NOPAD'
                yOut         = curves.ySGno;
                applyPadding = false;
                filterTag    = "SG";
            case 'MOVMEAN'
                yOut         = curves.yRApad;
                applyPadding = true;
                filterTag    = "MOVMEAN";
            case 'GAUSSIAN'
                yOut         = curves.yGaussPad;
                applyPadding = true;
                filterTag    = "GAUSSIAN";
        end

        accepted = true;
        uiresume(hFig);
    end

    function cancelCallback(~,~)
        accepted = false;
        uiresume(hFig);
    end
end

%======================================================================%
% Append applied filter settings into a TXT file in MAIN ROOT           %
%======================================================================%
function appendFilterSettingTxt(rootDir, fileKey, signalType, period, applyPadding, filterTag, headMode, tailMode, fitFrac, sourceMode)
    try
        outTxt = fullfile(rootDir, 'filter_settings_applied.txt');
        ts = datestr(now, 'yyyy-mm-dd HH:MM:SS');

        line = sprintf('[%s] source=%s | key=%s | signal=%s | period=%d | padding=%s | filter=%s | head=%s | tail=%s | fit_frac=%.4f\n', ...
            ts, char(sourceMode), char(string(fileKey)), char(string(signalType)), ...
            round(period), char(string(applyPadding)), char(string(filterTag)), ...
            char(string(headMode)), char(string(tailMode)), double(fitFrac));

        fid = fopen(outTxt, 'a');
        if fid ~= -1
            fprintf(fid, '%s', line);
            fclose(fid);
        end
    catch
        % ignore TXT logging errors
    end
end

%======================================================================%
% Helper to compute all candidate curves (used by both modes)           %
%======================================================================%
function curves = computeFilteredCurves(signal, freqRange, period, ...
                                        paddingHeadMode, paddingTailMode, fitFracTail)
    signal = signal(:)';
    freqRange = freqRange(:)';

    L = length(signal);

    period   = max(3, min(period, L-2));
    framelen = max(3, period + mod(period+1,2));

    if framelen > L
        framelen = 2*floor((L-1)/2) + 1;
        period   = framelen;
    end

    order           = 2;
    iters           = 10;
    averagingWindow = framelen;

    if strcmpi(paddingHeadMode,'mirror')
        yStart = signal(framelen:-1:1);
    else
        sineModel = @(b, x) b(1)*sin(b(2)*x + b(3)) + b(4);
        beta0     = [range(signal)/2, 2*pi/period, 0, mean(signal)];

        fitLen = max(3, floor(length(signal) * 0.7));
        x_fit  = 1:fitLen;
        y_fit  = signal(x_fit);

        opts = optimoptions('lsqcurvefit','Display','off');
        beta = lsqcurvefit(sineModel, beta0, x_fit, y_fit, [], [], opts);

        xPad  = -(framelen-1):0;
        yStart = sineModel(beta, xPad);
    end

    if strcmpi(paddingTailMode,'mirror')
        yEnd = signal(end:-1:end-framelen+1);
    else
        sineModel = @(b, x) b(1)*sin(b(2)*x + b(3)) + b(4);
        beta0     = [range(signal)/2, 2*pi/period, 0, mean(signal)];

        fitLen = max(3, floor(length(signal) * 0.7));
        x_fit  = 1:fitLen;
        y_fit  = signal(x_fit);

        opts = optimoptions('lsqcurvefit','Display','off');
        beta = lsqcurvefit(sineModel, beta0, x_fit, y_fit, [], [], opts);

        xEnd = length(signal) + (1:framelen);
        yEnd = sineModel(beta, xEnd);
    end

    padlength = length(yStart);
    yPadded   = [yStart, signal, yEnd];

    ySGpad = yPadded;
    for kk = 1:iters
        ySGpad = sgolayfilt(ySGpad, order, framelen);
    end
    yRApad    = movmean(ySGpad, averagingWindow, 'Endpoints','shrink');
    yGaussPad = smoothdata(ySGpad, 'gaussian', averagingWindow);

    ySGpad    = ySGpad(padlength+1:end-padlength);
    yRApad    = yRApad(padlength+1:end-padlength);
    yGaussPad = yGaussPad(padlength+1:end-padlength);

    ySGno = signal;
    for kk = 1:iters
        ySGno = sgolayfilt(ySGno, order, framelen);
    end
    yRAno    = movmean(ySGno, averagingWindow, 'Endpoints','shrink');
    yGaussNo = smoothdata(ySGno, 'gaussian', averagingWindow);

    ySGpad_ext    = extendTailLinear(freqRange, ySGpad,    fitFracTail);
    yRApad_ext    = extendTailLinear(freqRange, yRApad,    fitFracTail);
    yGaussPad_ext = extendTailLinear(freqRange, yGaussPad, fitFracTail);

    ySGno_ext     = extendTailLinear(freqRange, ySGno,     fitFracTail);
    yRAno_ext     = extendTailLinear(freqRange, yRAno,     fitFracTail);
    yGaussNo_ext  = extendTailLinear(freqRange, yGaussNo,  fitFracTail);

    curves = struct();
    curves.ySGpad    = ySGpad_ext(:);
    curves.yRApad    = yRApad_ext(:);
    curves.yGaussPad = yGaussPad_ext(:);
    curves.ySGno     = ySGno_ext(:);
    curves.yRAno     = yRAno_ext(:);
    curves.yGaussNo  = yGaussNo_ext(:);
end

function yExt = extendTailLinear(x, y, frac)
    yExt = y(:);
    x    = x(:);

    Lloc = numel(yExt);
    if Lloc < 3
        return;
    end

    fitLenLoc = max(3, floor(frac * Lloc));
    if fitLenLoc >= Lloc
        return;
    end

    i2 = fitLenLoc;
    i1 = max(1, i2 - 1);

    x1 = x(i1);  y1 = yExt(i1);
    x2 = x(i2);  y2 = yExt(i2);

    if x2 == x1
        m = 0;
    else
        m = (y2 - y1) / (x2 - x1);
    end

    for ii = fitLenLoc+1 : Lloc
        yExt(ii) = y2 + m * (x(ii) - x2);
    end
end

%======================================================================%
% Process band-level merged folders (JSON bands)                        %
%======================================================================%
function processBandLevelMerged( ...
        sampleName, ...
        methodFolderName, ...
        methodFolderPath, ...
        methodCfg, ...
        mergedSubFolderName, ...
        permMergedSuffix, ...
        permAveragedSuffix, ...
        s2pMergedSuffix, ...
        s2pAveragedSuffix, ...
        rootDir, ...
        filterMode)

    % Get band names from JSON
    bands = methodCfg.Bands;

    if isstruct(bands)
        bandNames = {bands.Name};
    elseif iscell(bands)
        bandNames = cellfun(@(x) x.Name, bands, 'UniformOutput', false);
    elseif isstring(bands)
        bandNames = cellstr(bands);
    elseif ischar(bands)
        bandNames = {bands};
    else
        error("Unsupported type for methodCfg.Bands: %s", class(bands));
    end

    allSubFolders = dir(fullfile(methodFolderPath, '*'));
    allSubFolders = allSubFolders([allSubFolders.isdir] & ...
                                  ~startsWith({allSubFolders.name}, '.'));

    bandFolders = allSubFolders(ismember({allSubFolders.name}, bandNames));

    for bIdx = 1:numel(bandFolders)
        bandFolderName = bandFolders(bIdx).name;
        bandFolderPath = fullfile(bandFolders(bIdx).folder, bandFolderName);

        mergedDirPath = fullfile(bandFolderPath, mergedSubFolderName);
        if ~exist(mergedDirPath, 'dir')
            continue;
        end

        filteredDir = fullfile(bandFolderPath, 'Merged_Filtered_Revision');
        if ~exist(filteredDir, 'dir')
            mkdir(filteredDir);
        end

        fprintf('\n=== Sample: %s | Method: %s | Band: %s ===\n', ...
                sampleName, methodFolderName, bandFolderName);
        fprintf('    Merged folder:   %s\n', mergedDirPath);
        fprintf('    Filtered output: %s\n', filteredDir);

        % Shared keys: ensures SAME filters for roughness/without-roughness in this folder
        fakeKeyEps = fullfile(mergedDirPath, 'FILTER_KEY_EPS.csv');
        fakeKeyTan = fullfile(mergedDirPath, 'FILTER_KEY_TAN.csv');

        csvFiles = dir(fullfile(mergedDirPath, '*.csv'));
        if isempty(csvFiles)
            fprintf('    No CSV files inside this merged folder.\n');
            continue;
        end

        for k = 1:numel(csvFiles)
            inName = csvFiles(k).name;
            inPath = fullfile(csvFiles(k).folder, inName);
            outPath = fullfile(filteredDir, inName);

            % 1) SKIP: base files (copy only, no filtering)
            isBase = endsWith(inName, permMergedSuffix,   'IgnoreCase', true) || ...
                     endsWith(inName, permAveragedSuffix, 'IgnoreCase', true) || ...
                     endsWith(inName, s2pMergedSuffix,    'IgnoreCase', true);

            if ~isempty(s2pAveragedSuffix)
                isBase = isBase || endsWith(inName, s2pAveragedSuffix, 'IgnoreCase', true);
            end

            if isBase
                fprintf('  [SKIP-FILTER] Copy base file: %s\n', inName);
                copyfile(inPath, outPath);
                continue;
            end

            % 2) Only focus on files that contain "Merged" or "Averaged"
            if ~(contains(inName, 'Merged') || contains(inName, 'Averaged'))
                fprintf('  [COPY ONLY] %s\n', inName);
                copyfile(inPath, outPath);
                continue;
            end

            % 3) Determine file type
            if contains(inName, 'S2PMerged', 'IgnoreCase', true)
                fileType = 's2p_merged';       % 3-col blocks: F, eps, tan
            elseif contains(inName, 'S2PAveraged', 'IgnoreCase', true)
                fileType = 's2p_averaged';     % averaged S2P
            elseif contains(inName, 'Merged', 'IgnoreCase', true)
                fileType = 'perm_merged';      % 4-col blocks: F, eps, tan, thk
            elseif contains(inName, 'Averaged', 'IgnoreCase', true)
                fileType = 'perm_averaged';    % averaged permittivity
            else
                fileType = 'other';
            end

            fprintf('  Processing (%s): %s\n', fileType, inName);

            % 4) Read, filter, write
            try
                T = readtable(inPath, 'VariableNamingRule', 'preserve');
                data = T{:,:};

                switch fileType
                    case 'perm_merged'
                        dataFiltered = filterPermMergedBlocks( ...
                            data, rootDir, fakeKeyEps, fakeKeyTan, filterMode);
                        Tout = array2table(dataFiltered, ...
                            'VariableNames', T.Properties.VariableNames);
                        writetable(Tout, outPath);

                    case 'perm_averaged'
                        Tfiltered = filterPermAveragedTable( ...
                            T, rootDir, fakeKeyEps, fakeKeyTan, filterMode);
                        writetable(Tfiltered, outPath);

                    case 's2p_merged'
                        dataFiltered = filterS2PMergedBlocks( ...
                            data, rootDir, fakeKeyEps, fakeKeyTan, filterMode);
                        Tout = array2table(dataFiltered, ...
                            'VariableNames', T.Properties.VariableNames);
                        writetable(Tout, outPath);

                    case 's2p_averaged'
                        Tfiltered = filterS2PAveragedTable( ...
                            T, rootDir, fakeKeyEps, fakeKeyTan, filterMode);
                        writetable(Tfiltered, outPath);

                    otherwise
                        copyfile(inPath, outPath);
                end

                close all; % prevent figure accumulation

            catch ME
                fprintf('    ! Error processing %s: %s\n', inName, ME.message);
                if ~isfile(outPath)
                    copyfile(inPath, outPath);
                end
            end
        end
    end
end

%======================================================================%
% 4-col block: [Freq, Eps, Tan, Thickness] × N  (permittivity merged)   %
%======================================================================%
function dataOut = filterPermMergedBlocks(data, rootDir, fakeKeyEps, fakeKeyTan, filterMode)
    [~, nCols] = size(data);
    if mod(nCols, 4) ~= 0
        warning('    ! perm_merged: number of columns is not multiple of 4. Leaving unchanged.');
        dataOut = data;
        return;
    end

    nBlocks = nCols / 4;
    freqHz  = data(:,1);
    freqGHz = freqHz ./ 1e9;

    dataOut = data;

    for b = 1:nBlocks
        baseC  = (b-1)*4;
        epsCol = baseC + 2;
        tanCol = baseC + 3;

        epsSignal = data(:, epsCol);
        tanSignal = data(:, tanCol);

        yEps = filterUoBPermittivity(freqGHz, epsSignal, fakeKeyEps, rootDir, filterMode);
        yTan = filterUoBPermittivity(freqGHz, tanSignal, fakeKeyTan, rootDir, filterMode);

        dataOut(:, epsCol) = yEps(:);
        dataOut(:, tanCol) = yTan(:);
    end
end
