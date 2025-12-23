function runAllMaterials()
% This function reads config_VNA.json and decides which file formats and algorithm
% should be processed for each material folder.

    configPath = 'config_VNA.json';
    if ~isfile(configPath)
        error('Could not find config file: %s', configPath);
    end

    text   = fileread(configPath);
    config = jsondecode(text);

    msg = "This function calculates permittivity and loss tangent for VNA measurements:" + newline + newline + ...
          "• either using GRL (Generalized Reflection Line)"  + newline + newline + ...
          "• or TRL (Thru-Reflect-Line)"  + newline + newline + "Click 'Continue' to proceed or 'Cancel' to exit.";

    userChoice = questdlg(msg, 'Measurement File Format Info', 'Continue', 'Cancel', 'Cancel');
    if ~strcmp(userChoice, 'Continue')
        disp('Operation cancelled by user.');
        return;
    end

    % Decide root directory
    useConfigDir = false;
    if isfield(config, "FilePatterns") && isfield(config.FilePatterns, 'rootDir') && ~isempty(config.FilePatterns.rootDir)
        if isfolder(config.FilePatterns.rootDir)
            useConfigDir = true;
        end
    end

    if useConfigDir
        rootDir = config.FilePatterns.rootDir;
        fprintf('Root folder is: %s\n', rootDir);
    else
        msg = 'Do you want to select a new root folder?';
        userChoice = questdlg(msg, 'Root Folder Selection', 'Continue', 'Cancel', 'Cancel');

        if ~strcmp(userChoice, 'Continue')
            disp('Operation cancelled by user.');
            return;
        end

        rootDir = uigetdir([], 'Select your root folder');
        if isequal(rootDir, 0)
            disp('Operation cancelled by user.');
            return;
        end
    end

    % Ask which input type to process
    dataChoice = questdlg('Input data type?', 'Input Source', 'CSV only', 'S2P only', 'CSV only');

    % Decide which layout / pattern to use
    switch dataChoice
        case 'CSV only'
            config.Settings.UseS2P = false;
            config.Debug.CompareCSVvsS2P = false;
            mergedPattern = config.FilePatterns.MergedFilePattern;                 % "*Merged_Data.csv"

        case 'S2P only'
            config.Settings.UseS2P       = true;
            config.Debug.CompareCSVvsS2P = false;
            mergedPattern = ['*' config.MergedProcessing.Output.S2PMergedSuffix];  % "*_S2PMerged.csv"

        otherwise
            disp('Operation cancelled by user.');
            return;
    end

    folders = dir(rootDir);
    folders = folders([folders.isdir] & ~startsWith({folders.name}, '.'));

    % Materials list from JSON
    materialsStruct = config.Materials;

    % Top-level folders, names must match config.Materials.Name
    for i = 1:length(folders)
        folderName  = folders(i).name;
        fullSubPath = fullfile(rootDir, folderName);

        matchIdx = find(arrayfun(@(x) strcmpi(x.Name, folderName), materialsStruct), 1);

        if isempty(matchIdx)
            fprintf('No match in config for folder: %s\n', folderName);
            continue;
        end

        % Thickness and roughness values from config
        matchedMat   = materialsStruct(matchIdx);
        thicknessVal = matchedMat.Thickness;
        roughnessVal = matchedMat.Roughness;

        mergedDirName = config.FilePatterns.MergedDirectory;
        mergedDirs    = dir(fullfile(fullSubPath, '**', mergedDirName));
        mergedDirs    = mergedDirs([mergedDirs.isdir]);

        for j = 1:length(mergedDirs)
            subFolder = fullfile(mergedDirs(j).folder, mergedDirs(j).name);

            % For CSV mode: "*Merged_Data.csv", For S2P mode: "*_S2PMerged.csv"
            mergedFiles = dir(fullfile(subFolder, mergedPattern));

            for k = 1:length(mergedFiles)
                filePath = fullfile(subFolder, mergedFiles(k).name);

                % GRL or TRL algorithm selection
                if isfield(config, "Settings") && isfield(config.Settings, "forceAlgorithm") && ...
                   ~isempty(config.Settings.forceAlgorithm) && ...
                   (strcmpi(config.Settings.forceAlgorithm, 'GRL') || strcmpi(config.Settings.forceAlgorithm, 'TRL'))

                    localAlgorithm = upper(config.Settings.forceAlgorithm);
                else
                    if contains(subFolder, config.Settings.GRLFolderKeyword, 'IgnoreCase', true)
                        localAlgorithm = 'GRL';
                    else
                        localAlgorithm = 'TRL';
                    end
                end

                fprintf('Processing: %s | Algorithm: %s\n', filePath, localAlgorithm);

                % Main processing function
                processMergedFiles(filePath, config, thicknessVal, roughnessVal, localAlgorithm);
            end
        end
    end
end

function processMergedFiles(filePath, config, thickness, roughness, algorithmType)
% Prepares merged data for the core extraction algorithm and saves results.
%
% CSV permittivity merged files:
%   - Uses config.DataLayoutCSV (BlockSize = 4, Columns.Frequency/Epsilon/TanDelta/Thickness)
%   - Each block (4 columns) is [f, eps, tan, thk]
%
% S2P merged files:
%   - Uses config.DataLayoutS2PCSV (BlockSize = 9)
%   - Each block (9 columns) is:
%     [f, S11_mag, S11_phase, S21_mag, S21_phase, S12_mag, S12_phase, S22_mag, S22_phase]

    [folder, baseFileName, ~] = fileparts(filePath);
    fullData   = readmatrix(filePath);
    cellData   = readcell(filePath);
    headerData = cellData(1, :);

    [~, numCols] = size(fullData);

    % Thickness / roughness scaling to meters
    if strcmpi(config.Units.ThicknessUnit, 'mm')
        unitScaleThk = 1E-3;
    elseif strcmpi(config.Units.ThicknessUnit, 'um')
        unitScaleThk = 1E-6;
    else
        unitScaleThk = 1;
    end

    if strcmpi(config.Units.RoughnessUnit, 'um')
        unitScaleRough = 1E-6;
    elseif strcmpi(config.Units.RoughnessUnit, 'mm')
        unitScaleRough = 1E-3;
    else
        unitScaleRough = 1;
    end

    sampleThickness = thickness * unitScaleThk;
    sigma           = roughness * unitScaleRough;

    % Decide mode: CSV permittivity vs S2P
    useS2P = isfield(config, 'Settings') && isfield(config.Settings, 'UseS2P') && config.Settings.UseS2P;

    if ~useS2P
        %==============================================================
        % CSV PERMITTIVITY MODE  (DataLayoutCSV)
        %==============================================================
        if ~isfield(config, 'DataLayoutCSV')
            error('config.DataLayoutCSV is not defined for CSV mode.');
        end

        blockSize = config.DataLayoutCSV.BlockSize;
        numBlocks = floor(numCols / blockSize);

        newData  = fullData;   % with roughness
        newData1 = fullData;   % without roughness

        % Column indices (0-based offset because of block)
        colFreq = config.DataLayoutCSV.Columns.Frequency - 1;
        colEps  = config.DataLayoutCSV.Columns.Epsilon   - 1;
        colTan  = config.DataLayoutCSV.Columns.TanDelta  - 1;

        for k = 1:numBlocks
            baseIdx = (k - 1) * blockSize + 1;

            freq     = fullData(:, baseIdx + colFreq);
            epsilon  = fullData(:, baseIdx + colEps);
            tandelta = fullData(:, baseIdx + colTan);

            tempData      = [freq, epsilon, tandelta];
            validRows     = ~any(isnan(tempData), 2);
            tempDataValid = tempData(validRows, :);

            % Configuration for this run
            currentConfig = config;
            currentConfig.CurrentSigma = 0;
            currentConfig.CurrentAlgo  = algorithmType;
            currentConfig.Debug.FileLabel = filePath;

            % Core extraction (no roughness first)
            % We use the no-roughness case as an initial-condition predictor for the roughness case
            resultsTable = extractEpsTandelta(tempDataValid, sampleThickness, currentConfig);

            if config.Debug.PauseDuration > 0
                pause(config.Debug.PauseDuration);
            end

            if config.Debug.SaveWithoutRoughness
                newData1(validRows, baseIdx + colEps) = resultsTable.EpsilonReal;
                newData1(validRows, baseIdx + colTan) = resultsTable.TanDelta;
            end

            % If roughness enabled
            if config.Debug.EnableRoughness
                tempData2 = [resultsTable.Frequency_Hz, resultsTable.EpsilonReal, resultsTable.TanDelta];
                currentConfig.CurrentSigma = sigma;

                resultsTable = extractEpsTandelta(tempData2, sampleThickness, currentConfig);

                % Results with roughness
                newData(validRows, baseIdx + colEps) = resultsTable.EpsilonReal;
                newData(validRows, baseIdx + colTan) = resultsTable.TanDelta;

                if config.Debug.PauseDuration > 0
                    pause(config.Debug.PauseDuration);
                end
            end
        end

        %------------------------ Save (CSV mode) ------------------------
        if config.Debug.SaveWithoutRoughness
            suffix1 = config.FilePatterns.OutputFileSuffix1;
            if isempty(suffix1), suffix1 = '_withoutRoughness'; end

            outFile = fullfile(folder, [baseFileName, suffix1, '.csv']);
            writetable(array2table(newData1, 'VariableNames', headerData), outFile);
            fprintf("Saved (no roughness): %s\n", outFile);
        end

        if config.Debug.SaveWithRoughness
            suffix = config.FilePatterns.OutputFileSuffix;
            if isempty(suffix), suffix = '_Roughness'; end

            outFile = fullfile(folder, [baseFileName, suffix, '.csv']);
            writetable(array2table(newData, 'VariableNames', headerData), outFile);
            fprintf("Saved (with roughness): %s\n", outFile);
        end

    else
        %==============================================================
        % S2P MODE  (DataLayoutS2PCSV, BlockSize = 9)
        %==============================================================
        if ~isfield(config, 'DataLayoutS2PCSV')
            error('config.DataLayoutS2PCSV is not defined for S2P mode.');
        end

        blockSize = config.DataLayoutS2PCSV.BlockSize;   % should be 9
        numBlocks = floor(numCols / blockSize);

        % We will build two output matrices:
        %  - mergedNoRoughS2P : [Freq, EpsReal, TanDelta] per block (σ=0)
        %  - mergedRoughS2P   : same with roughness (σ>0), if enabled
        mergedNoRoughS2P = [];
        mergedRoughS2P   = [];
        blockCount       = 0;

        for k = 1:numBlocks
            baseIdx = (k - 1) * blockSize + 1;

            % Extract the 9-column block
            freq      = fullData(:, baseIdx + 0);
            S11_mag   = fullData(:, baseIdx + 1);
            S11_phase = fullData(:, baseIdx + 2);
            S21_mag   = fullData(:, baseIdx + 3);
            S21_phase = fullData(:, baseIdx + 4);
            S12_mag   = fullData(:, baseIdx + 5);
            S12_phase = fullData(:, baseIdx + 6);
            S22_mag   = fullData(:, baseIdx + 7);
            S22_phase = fullData(:, baseIdx + 8);

            blockData = [freq, S11_mag, S11_phase, S21_mag, S21_phase, S12_mag, S12_phase, S22_mag, S22_phase];

            % Remove rows with any NaN in the block
            validRows = ~any(isnan(blockData), 2);
            blockData = blockData(validRows, :);

            if isempty(blockData)
                continue;
            end

            blockCount = blockCount + 1;

            freq_valid  = blockData(:,1);

            % Convert mag/phase → complex S-parameters
            ph2rad = pi/180;
            S11 = blockData(:,2) .* exp(1i * blockData(:,3) * ph2rad);
            S21 = blockData(:,4) .* exp(1i * blockData(:,5) * ph2rad);
            S12 = blockData(:,6) .* exp(1i * blockData(:,7) * ph2rad);
            S22 = blockData(:,8) .* exp(1i * blockData(:,9) * ph2rad);

            % Build Scombined struct
            Scombined.freq = freq_valid;
            Scombined.S11  = S11;
            Scombined.S21  = S21;
            Scombined.S12  = S12;
            Scombined.S22  = S22;

            % We use the no-roughness case as an initial-condition predictor for the roughness case
            %=================== No roughness (σ = 0) ===================
            currentConfig              = config;
            currentConfig.CurrentSigma = 0;
            currentConfig.CurrentAlgo  = algorithmType;
            currentConfig.Debug.FileLabel = filePath;

            resultsNoRough = extractEpsTandelta(Scombined, sampleThickness, currentConfig);
            blockNoRough = table2array(resultsNoRough(:, {'Frequency_Hz','EpsilonReal','TanDelta'}));

            % Pad & append into mergedNoRoughS2P
            mergedNoRoughS2P = padAndAppendBlock(mergedNoRoughS2P, blockNoRough);

            if config.Debug.PauseDuration > 0
                pause(config.Debug.PauseDuration);
            end

            %===================== With roughness (σ > 0) =================
            if config.Debug.EnableRoughness
                currentConfig.CurrentSigma = sigma;
                initData = [resultsNoRough.Frequency_Hz, resultsNoRough.EpsilonReal, resultsNoRough.TanDelta];

                resultsRough = extractEpsTandelta(Scombined, sampleThickness, currentConfig, initData);
                blockRough = table2array(resultsRough(:, {'Frequency_Hz','EpsilonReal','TanDelta'}));

                mergedRoughS2P = padAndAppendBlock(mergedRoughS2P, blockRough);
                if config.Debug.PauseDuration > 0
                    pause(config.Debug.PauseDuration);
                end
            end
        end

        if blockCount == 0
            fprintf('No valid S2P blocks in %s\n', filePath);
            return;
        end

        %------------------------ Save (S2P mode) ------------------------

        % Column names for [Freq, EpsReal, TanDelta] per block
        s2pResultHeaders = {'Frequency_Hz', 'EpsilonReal', 'TanDelta'};

        % Save without roughness
        if config.Debug.SaveWithoutRoughness && ~isempty(mergedNoRoughS2P)
            nCols = size(mergedNoRoughS2P, 2);
            nBlocksEffective = nCols / numel(s2pResultHeaders);

            colLabelsNR = strings(1, nCols);
            idx = 1;
            for bIdx = 1:nBlocksEffective
                for h = 1:numel(s2pResultHeaders)
                    colLabelsNR(idx) = sprintf('%d_%s', bIdx, s2pResultHeaders{h});
                    idx = idx + 1;
                end
            end

            suffix1S2P = '';
            if isfield(config.FilePatterns, 'OutputFileSuffix1S2P')
                suffix1S2P = config.FilePatterns.OutputFileSuffix1S2P;
            end
            if isempty(suffix1S2P)
                suffix1S2P = '_withoutRoughnessS2P';
            end

            outFile1 = fullfile(folder, [baseFileName, suffix1S2P, '.csv']);
            writetable(array2table(mergedNoRoughS2P, 'VariableNames', cellstr(colLabelsNR)), outFile1);
            fprintf("Saved S2P (no roughness): %s\n", outFile1);
        end

        % Save with roughness
        if config.Debug.SaveWithRoughness && ~isempty(mergedRoughS2P)
            nCols = size(mergedRoughS2P, 2);
            nBlocksEffective = nCols / numel(s2pResultHeaders);

            colLabelsR = strings(1, nCols);
            idx = 1;
            for bIdx = 1:nBlocksEffective
                for h = 1:numel(s2pResultHeaders)
                    colLabelsR(idx) = sprintf('%d_%s', bIdx, s2pResultHeaders{h});
                    idx = idx + 1;
                end
            end

            suffixS2P = '';
            if isfield(config.FilePatterns, 'OutputFileSuffixS2P')
                suffixS2P = config.FilePatterns.OutputFileSuffixS2P;
            end
            if isempty(suffixS2P)
                suffixS2P = '_RoughnessS2P';
            end

            outFile2 = fullfile(folder, [baseFileName, suffixS2P, '.csv']);
            writetable(array2table(mergedRoughS2P, 'VariableNames', cellstr(colLabelsR)), outFile2);
            fprintf("Saved S2P (with roughness): %s\n", outFile2);
        end
    end

    close all;
end

% Helper for padding and horizontal concatenation
function merged = padAndAppendBlock(merged, block)
    if isempty(merged)
        merged = block;
        return;
    end

    nRowsBlock   = size(block, 1);
    nRowsMerged  = size(merged, 1);
    maxRows      = max(nRowsBlock, nRowsMerged);

    if nRowsMerged < maxRows
        merged(end+1:maxRows, :) = NaN;
    end
    if nRowsBlock < maxRows
        block(end+1:maxRows, :) = NaN;
    end

    merged = [merged block];
end

function [resultsTable, Scombined] = extractEpsTandelta(data, sampleThickness, config, initData)
% Unified extraction:
%   - If "data" is a numeric matrix [freq, eps, tan], it uses the CSV workflow.
%   - If "data" is a struct with S-parameters (freq, S11, S21, S12, S22) it uses the S2P workflow.

    warning off;

    if nargin < 4
        initData = [];
    end

    c0 = config.Settings.C0;
    sigma = config.CurrentSigma;
    algoType = config.CurrentAlgo;      % 'GRL' or 'TRL'

    isSpecialBand = false;
    if isfield(config, 'Optimization') && isfield(config.Optimization, 'SpecialBandBounds')
        sb = config.Optimization.SpecialBandBounds;

        if isfield(sb, 'Enable') && sb.Enable ...
           && isfield(sb, 'BandNames') && iscell(sb.BandNames) ...
           && isfield(config, 'Debug') && isfield(config.Debug, 'FileLabel')

            fileLabel = config.Debug.FileLabel;

            for nn = 1:numel(sb.BandNames)
                if contains(fileLabel, sb.BandNames{nn}, 'IgnoreCase', true)
                    isSpecialBand = true;
                    break;
                end
            end
        end
    end

    % Detect input type: S2P struct vs CSV-like numeric
    if isstruct(data)
        %============================== S2P BRANCH ==============================
        Scombined = data;

        freq = Scombined.freq(:);
        omega = 2 * pi .* freq;
        numPoints = numel(freq);

        % --- Initial guess selection for S2P ---
        if sigma == 0 || isempty(initData)
            % No roughness: initial guess from measured S21
            [epsInitial, tanDeltaInitial] = initialGuessFromS21(freq, Scombined.S21(:), sampleThickness, c0);
        else
            % Roughness case: use initData as initial guess
            epsInitial = initData(:,2);
            tanDeltaInitial = initData(:,3);

            % Safety check
            if numel(epsInitial) ~= numPoints || numel(tanDeltaInitial) ~= numPoints
                warning('initData size mismatch, falling back to initialGuessFromS21.');
                [epsInitial, tanDeltaInitial] = initialGuessFromS21(freq, Scombined.S21(:), sampleThickness, c0);
            end
        end

        % For plotting
        if strcmpi(algoType, 'TRL')
            S_plot = Scombined;         % full struct
        else
            S_plot = Scombined.S21(:);  % combined S vector for GRL plots
        end

        S_obj = Scombined; % needed for objective

    else
        %============================== CSV BRANCH ==============================
        csvData   = data;
        freq      = csvData(:,1);
        omega     = 2 * pi .* freq;
        numPoints = numel(freq);

        epsCol = csvData(:,2);
        tanCol = csvData(:,3);

        if strcmpi(algoType, 'GRL')
            % GRL from CSV
            epsComplex = epsCol - 1i .* tanCol .* epsCol;
            sqrtEps    = sqrt(epsComplex);

            Scombined = epsComplex ./ ( ...
                epsComplex .* cos(omega .* sqrtEps .* sampleThickness ./ c0) + ...
                1i .* sqrtEps .* 0.5 .* (1 + epsComplex) .* ...
                sin(omega .* sqrtEps .* sampleThickness ./ c0));

            rawPhase  = unwrap(angle(Scombined));
            Scombined = abs(Scombined) .* exp(1i * rawPhase);

            S_plot = Scombined;  % vector
            S_obj  = Scombined;  % objective uses the combined S vector

            if sigma == 0
                % No roughness: initial guess from this combined S
                [epsInitial, tanDeltaInitial] = initialGuessFromS21(freq, Scombined(:), sampleThickness, c0);
            else
                % Roughness: epsCol / tanCol already no-roughness result
                epsInitial = epsCol;
                tanDeltaInitial = tanCol;
            end

        elseif strcmpi(algoType, 'TRL')
            % TRL from CSV
            SmodelArray = arrayfun(@(k) modelSparams(epsCol(k), tanCol(k), omega(k), sampleThickness, 0, 0, c0), 1:numPoints).';

            Scombined.freq = freq;
            Scombined.S11  = [SmodelArray.S11].';
            Scombined.S21  = [SmodelArray.S21].';
            Scombined.S12  = [SmodelArray.S12].';
            Scombined.S22  = [SmodelArray.S22].';

            S_plot = Scombined;
            S_obj  = Scombined;

            if sigma == 0
                % No roughness: initial guess from synthetic S21
                [epsInitial, tanDeltaInitial] = initialGuessFromS21(freq, Scombined.S21(:), sampleThickness, c0);
            else
                % Roughness: epsCol / tanCol already no-roughness result
                epsInitial = epsCol;
                tanDeltaInitial = tanCol;
            end
        else
            error('Unknown algorithm type: %s', algoType);
        end
    end

    optionsFmincon = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'interior-point');

    epsExtracted = zeros(numPoints,1);
    tanExtracted = zeros(numPoints,1);

    lb_plot = zeros(numPoints, 2);
    ub_plot = zeros(numPoints, 2);

    if strcmpi(algoType, 'GRL')
        flagA = 0;
    else
        flagA = 1;
    end

    bf = config.Optimization.BoundFactor;

    for k = 1:numPoints
        eps0 = epsInitial(k);
        tan0 = max(tanDeltaInitial(k), 1e-5);

        % ----------------- Base bounds (same logic as before) -----------------
        if ~isSpecialBand
            lb_eps = eps0 * (1 - bf);
            ub_eps = eps0 * (1 + bf);

            scale  = config.Optimization.SpecialBandBounds.TanScaleFactor;
            lb_tan = -scale * tan0;
            ub_tan =  scale * tan0;
        else
            lb_eps = eps0 * (1 - 4*bf);
            ub_eps = eps0 * (1 + 4*bf);

            lb_tan = -1;
            ub_tan =  1;
        end

        lb = [lb_eps; lb_tan];
        ub = [ub_eps; ub_tan];

        lb_plot(k,:) = lb(:).';
        ub_plot(k,:) = ub(:).';

        % ----------------- Objective (same as before) -----------------
        if strcmpi(algoType, 'GRL')
            if isstruct(S_obj)
                S_meas_k = S_obj.S21(k);
            else
                S_meas_k = S_obj(k);
            end
            objective = @(x) sum(optimizationErrorGRLalg( ...
                x, omega(k), sampleThickness, S_meas_k, flagA, sigma, c0).^2 );
        else
            objective = @(x) sum(optimizationErrorTRLalg( ...
                x, omega(k), sampleThickness, ...
                S_obj.S11(k), S_obj.S12(k), ...
                S_obj.S21(k), S_obj.S22(k), ...
                flagA, sigma, c0).^2 );
        end

        initialConds       = [eps0, tan0];
        extractedVals      = fmincon(objective, initialConds, [], [], [], [], lb, ub, [], optionsFmincon);
        epsExtracted(k)    = extractedVals(1);
        tanExtracted(k)    = extractedVals(2);
    end

    % Interactive plotting + optional recalculation
    if isfield(config, 'Debug') && config.Debug.EnablePlotting
        if isfield(config.Debug, 'FileLabel')
            fileLabel = config.Debug.FileLabel;
        else
            fileLabel = '';
        end

        [epsExtracted, tanExtracted] = plotResults( ...
            freq, epsExtracted, tanExtracted, epsInitial, tanDeltaInitial, ...
            lb_plot, ub_plot, S_plot, S_obj, sampleThickness, sigma, algoType, c0, fileLabel, ...
            isSpecialBand, bf, optionsFmincon, config);
    end

    resultsTable = table(freq, epsExtracted, tanExtracted, 'VariableNames', {'Frequency_Hz', 'EpsilonReal', 'TanDelta'});
end

%% --- Helper Functions ---
function errorVec = optimizationErrorGRLalg(initialConds, omega, thickness, S21, flagA, sigma, c0)
    epsComplex = initialConds(1) - 1j .* initialConds(1) .* initialConds(2);
    sqrtEps = sqrt(epsComplex);

    if flagA == 0
        f1Model = S21 .* (epsComplex .* cos(omega .* sqrtEps .* thickness ./ c0) + ...
            1i .* sqrtEps .* 0.5 .* (1 + epsComplex) .* sin(omega .* sqrtEps .* thickness ./ c0)) - epsComplex;
    else
        A = exp( 0.5 .* epsComplex .* omega.^2 .* sigma.^2 ./ (c0.^2)) .* (1 + sqrtEps).^2;
        B = exp(-0.5 .* epsComplex .* omega.^2 .* sigma.^2 ./ (c0.^2)) .* (1 - sqrtEps).^2;

        f1Model = S21 .* sqrtEps .* 0.25 .* (cos(omega .* sqrtEps .* thickness ./ c0) .* (A - B) + ...
            1i .* sin(omega .* sqrtEps .* thickness ./ c0) .* (A + B)) - epsComplex;
    end
    errorVec = [real(f1Model); imag(f1Model)];
end

function errorVec = optimizationErrorTRLalg(initialConds, omega, thickness, S11, S12, S21, S22, flagA, sigma, c0)
    epsComplex = initialConds(1) - 1j .* initialConds(2) .* initialConds(1);
    sqrtEps = sqrt(epsComplex);
    gamma = (1 - sqrtEps) ./ (1 + sqrtEps);

    if flagA == 0
        P = exp(-1j .* (omega ./ c0) .* sqrtEps .* thickness);
    else
        P = exp(-1j .* (omega ./ c0) .* sqrtEps .* thickness) .* exp(-0.5 .* omega.^2 .* epsComplex .* sigma.^2 ./ (2 .* c0.^2));
    end

    f1Model = (P.^2 - gamma.^2) ./ (1 - gamma.^2 .* P.^2);
    f2Model = (P .* (1 - gamma.^2) ./ (1 - gamma.^2 .* P.^2)) .* exp(1j * (omega ./ c0) .* thickness);

    f1Measured = S21 .* S12 - S11 .* S22;
    f2Measured = S21 ./ exp(-1j .* (omega / c0) .* thickness);

    errorVec = [
        real(f1Measured - f1Model);
        imag(f1Measured - f1Model);
        real(f2Measured - f2Model);
        imag(f2Measured - f2Model)];
end

function S = modelSparams(epsReal, tanDelta, omega, L, flagA, sigma, c0)
    epsr = epsReal - 1j .* tanDelta .* epsReal;
    sqrtEps = sqrt(epsr);
    gamma = (1 - sqrtEps) ./ (1 + sqrtEps);

    if flagA == 0 && sigma == 0
         P = exp(-1j .* (omega ./ c0) .* sqrtEps .* L);
    else
         P = exp(-1j .* (omega ./ c0) .* sqrtEps .* L) .* exp(-0.5 .* omega.^2 .* epsr .* sigma.^2 ./ (2 .* c0.^2));
    end

    S.S11 = gamma .* (1 - P.^2) ./ (1 - gamma.^2 .* P.^2);
    S.S21 = P .* (1 - gamma.^2) ./ (1 - gamma.^2 .* P.^2);
    S.S22 = S.S11;
    S.S12 = S.S21;
end

function [epsOut, tanOut] = plotResults( ...
    freq, epsExt, tanExt, epsInit, tanInit, lb, ub, S_measured, S_obj, thickness, sigma, algoType, c0, fileLabel, ...
    isSpecialBand, bf, optionsFmincon, config)
% Interactive plot with:
% - Left vertical sliders (centered): n bounds + n y-range
% - Right vertical sliders (centered): k bounds + k y-range
% - k bound max adjustable down to 0.001
% - k y-max adjustable down to 0.01
% - Recalculate button -> recompute eps/tan using current n/k bound settings
% - Save button (green, top-right) -> accept current results and close (then caller saves data)
% - Print button -> apply styling and save PDF into folder

    epsOut = epsExt(:);
    tanOut = tanExt(:);

    if nargin < 14
        fileLabel = '';
    end

    f_GHz = freq(:) ./ 1e9;
    omega = 2 * pi .* freq(:);

    % Convert eps/tan to n/k for plotting (small-loss approximation for k)
    nExt  = sqrt(max(epsOut,  0));
    nInit = sqrt(max(epsInit(:), 0));

    kExt  = 0.5 .* nExt  .* abs(tanOut);
    kInit = 0.5 .* nInit .* abs(tanInit(:));

    % Default bounds derived from eps/tan bounds
    n_lb0 = sqrt(max(lb(:,1), 0));
    n_ub0 = sqrt(max(ub(:,1), 0));

    tanAbsBound = max(abs(lb(:,2)), abs(ub(:,2)));
    k_ub0 = 0.5 .* sqrt(max(ub(:,1), 0)) .* tanAbsBound;
    k_lb0 = zeros(size(k_ub0));   % k lower bound shown as 0 axis

    baseKMax = max([kExt; kInit; k_ub0]);
    if ~isfinite(baseKMax) || baseKMax <= 0
        baseKMax = 1;
    end

    baseKBoundMax0 = max(k_ub0);
    if ~isfinite(baseKBoundMax0) || baseKBoundMax0 <= 0
        baseKBoundMax0 = max(baseKMax, 1);
    end

    % Nominal n reference for y-limits
    nRef = median(nInit(~isnan(nInit)));
    if isempty(nRef) || ~isfinite(nRef) || nRef <= 0
        nRef = 1;
    end

    % Default n y-limit percent (clamped to 1%..50%)
    tmpYL = [min(nExt), max(nExt)];
    p0 = max( abs(tmpYL - nRef) ./ nRef );
    p0 = min(max(p0, 0.01), 0.50);

    % Default k y-max (absolute)
    kY0 = 1.05 * baseKMax;
    kY0 = max(kY0, 0.01);

    % Font settings
    fontName = 'Times New Roman';
    fontSize = 12;

    % Create figure
    fig = figure('Name', ['Extraction and S-Parameters: ' algoType], 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
    set(fig, 'CloseRequestFcn', @onClose);

    % Subplot (1,1): n
    axN = subplot(3,2,1);
    hold on;
    pN = fill([f_GHz; flipud(f_GHz)], [n_lb0; flipud(n_ub0)], [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    lnNext  = plot(f_GHz, nExt,  'r',   'LineWidth', 1.5);
    lnNinit = plot(f_GHz, nInit, 'g--', 'LineWidth', 1.2);
    title(['Refractive Index (n), Algorithm: ' algoType], 'FontName', fontName, 'FontSize', fontSize);
    ylabel('n', 'FontName', fontName, 'FontSize', fontSize);
    legend({'Bounds', 'Extracted', 'Initial guess'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
    grid on;
    set(gca, 'FontName', fontName, 'FontSize', fontSize);

    % Subplot (1,2): k
    axK = subplot(3,2,2);
    hold on;
    pK = fill([f_GHz; flipud(f_GHz)], [k_lb0; flipud(k_ub0)], [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    lnKext  = plot(f_GHz, kExt,  'r',   'LineWidth', 1.5);
    lnKinit = plot(f_GHz, kInit, 'g--', 'LineWidth', 1.2);
    title('Extinction Coefficient (k)', 'FontName', fontName, 'FontSize', fontSize);
    ylabel('k', 'FontName', fontName, 'FontSize', fontSize);
    xlabel('Frequency (GHz)', 'FontName', fontName, 'FontSize', fontSize);
    legend({'Bounds', 'Extracted', 'Initial guess'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
    grid on;
    set(gca, 'FontName', fontName, 'FontSize', fontSize);
    ylim(axK, [0, kY0]);  % k y-min is always 0

    % --- S-parameter comparison plots ---
    isTRL = strcmpi(algoType, 'TRL') && isstruct(S_measured) && ...
            isfield(S_measured,'S11') && isfield(S_measured,'S21') && ...
            isfield(S_measured,'S12') && isfield(S_measured,'S22');

    axS = gobjects(4,1);
    lnMeas = gobjects(4,1);
    lnMod  = gobjects(4,1);

    if isTRL
        flagA = sigma > 0;

        S11_meas = S_measured.S11(:);
        S21_meas = S_measured.S21(:);
        S12_meas = S_measured.S12(:);
        S22_meas = S_measured.S22(:);

        subplot(3,2,3); axS(1)=gca; hold on;
        lnMeas(1) = plot(f_GHz, 20*log10(abs(S11_meas)), 'b',  'LineWidth', 1.5);
        lnMod(1)  = plot(f_GHz, nan(size(f_GHz)),       'r--','LineWidth', 1.5);
        title('|S_{11}| (dB)', 'FontName', fontName, 'FontSize', fontSize);
        ylabel('Magnitude (dB)', 'FontName', fontName, 'FontSize', fontSize);
        legend({'Measured', 'Model'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
        grid on; set(gca, 'FontName', fontName, 'FontSize', fontSize);

        subplot(3,2,4); axS(2)=gca; hold on;
        lnMeas(2) = plot(f_GHz, 20*log10(abs(S21_meas)), 'b',  'LineWidth', 1.5);
        lnMod(2)  = plot(f_GHz, nan(size(f_GHz)),       'r--','LineWidth', 1.5);
        title('|S_{21}| (dB)', 'FontName', fontName, 'FontSize', fontSize);
        ylabel('Magnitude (dB)', 'FontName', fontName, 'FontSize', fontSize);
        legend({'Measured', 'Model'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
        grid on; set(gca, 'FontName', fontName, 'FontSize', fontSize);

        subplot(3,2,5); axS(3)=gca; hold on;
        lnMeas(3) = plot(f_GHz, 20*log10(abs(S12_meas)), 'b',  'LineWidth', 1.5);
        lnMod(3)  = plot(f_GHz, nan(size(f_GHz)),       'r--','LineWidth', 1.5);
        title('|S_{12}| (dB)', 'FontName', fontName, 'FontSize', fontSize);
        ylabel('Magnitude (dB)', 'FontName', fontName, 'FontSize', fontSize);
        xlabel('Frequency (GHz)', 'FontName', fontName, 'FontSize', fontSize);
        legend({'Measured', 'Model'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
        grid on; set(gca, 'FontName', fontName, 'FontSize', fontSize);

        subplot(3,2,6); axS(4)=gca; hold on;
        lnMeas(4) = plot(f_GHz, 20*log10(abs(S22_meas)), 'b',  'LineWidth', 1.5);
        lnMod(4)  = plot(f_GHz, nan(size(f_GHz)),       'r--','LineWidth', 1.5);
        title('|S_{22}| (dB)', 'FontName', fontName, 'FontSize', fontSize);
        ylabel('Magnitude (dB)', 'FontName', fontName, 'FontSize', fontSize);
        xlabel('Frequency (GHz)', 'FontName', fontName, 'FontSize', fontSize);
        legend({'Measured', 'Model'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
        grid on; set(gca, 'FontName', fontName, 'FontSize', fontSize);

    else
        subplot(3,2,3); axS(1)=gca; hold on;
        lnMeas(1) = plot(f_GHz, 20*log10(abs(S_measured(:))), 'b',  'LineWidth', 1.5);
        lnMod(1)  = plot(f_GHz, nan(size(f_GHz)),            'r--','LineWidth', 1.5);
        title('Combined |S| (dB)', 'FontName', fontName, 'FontSize', fontSize);
        ylabel('Magnitude (dB)', 'FontName', fontName, 'FontSize', fontSize);
        legend({'Measured', 'Model'}, 'Location', 'best', 'FontName', fontName, 'FontSize', fontSize);
        grid on; set(gca, 'FontName', fontName, 'FontSize', fontSize);

        subplot(3,2,4); cla; axis off;
        text(0.5, 0.5, 'No full S-parameters available for GRL', 'HorizontalAlignment', 'center', 'FontName', fontName, 'FontSize', fontSize);

        subplot(3,2,5); cla; axis off;
        subplot(3,2,6); cla; axis off;
    end

    if ~isempty(fileLabel)
        annotation('textbox', [0 0.01 1 0.04], ...
            'String', fileLabel, ...
            'Interpreter', 'none', ...
            'HorizontalAlignment', 'center', ...
            'Color', 'w', ...
            'BackgroundColor', [0.7 0 0], ...
            'FontName', 'Times New Roman', ...
            'FontSize', 12, ...
            'FontAngle', 'italic', ...
            'EdgeColor', 'none');
    end

    % Leave space on left/right for vertical sliders
    all_axes = findall(fig, 'type', 'axes');
    for aa = 1:numel(all_axes)
        pos = get(all_axes(aa), 'Position');
        pos(1) = pos(1) + 0.045;
        pos(3) = pos(3) - 0.09;
        set(all_axes(aa), 'Position', pos);
    end

    %========================  GUI CONTROLS  ========================

    % LEFT (n): bounds scale + y-range
    uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.005 0.78 0.08 0.03], ...
        'String', 'n bounds', 'HorizontalAlignment', 'left', 'FontName', fontName, 'FontSize', 10);

    sNBound = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', 'Position', [0.015 0.25 0.015 0.5], ...
        'Min', 0.01, 'Max', 3.0, 'Value', 1.0, 'Callback', @onSliderChange);

    uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.032 0.78 0.08 0.03], ...
        'String', 'n y', 'HorizontalAlignment', 'left', 'FontName', fontName, 'FontSize', 10);

    sNY = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', 'Position', [0.042 0.25 0.015 0.5], ...
        'Min', 0.01, 'Max', 0.50, 'Value', p0, 'Callback', @onSliderChange);

    % RIGHT (k): bounds max (absolute) + y-max (absolute)
    uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.90 0.78 0.10 0.03], ...
        'String', 'k bounds', 'HorizontalAlignment', 'left', 'FontName', fontName, 'FontSize', 10);

    kBoundMaxMin = 0.001;
    kBoundMaxMax = max(3.0 * baseKBoundMax0, kBoundMaxMin);
    sKBoundAbs = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', 'Position', [0.94 0.25 0.015 0.5], ...
        'Min', kBoundMaxMin, 'Max', kBoundMaxMax, 'Value', min(max(baseKBoundMax0, kBoundMaxMin), kBoundMaxMax), 'Callback', @onSliderChange);

    uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', 'Position', [0.955 0.78 0.10 0.03], ...
        'String', 'k y', 'HorizontalAlignment', 'left', 'FontName', fontName, 'FontSize', 10);

    kYMin = 0.01;
    kYMax = max(3.0 * baseKMax, kYMin);
    sKYAbs = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', 'Position', [0.965 0.25 0.015 0.5], ...
        'Min', kYMin, 'Max', kYMax, 'Value', min(max(kY0, kYMin), kYMax), 'Callback', @onSliderChange);

    % TOP-RIGHT: Recalculate + Save (green) + Print
    uicontrol('Parent', fig, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.70 0.93 0.12 0.05], ...
        'String', 'Recalculate', 'FontName', fontName, 'FontSize', 12, 'Callback', @onRecalculate);

    btnSave = uicontrol('Parent', fig, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.84 0.93 0.07 0.05], ...
        'String', 'Save', 'FontName', fontName, 'FontSize', 12, 'Callback', @onSave);

    set(btnSave, 'BackgroundColor', [0.2 0.8 0.2], 'ForegroundColor', 'w');

    uicontrol('Parent', fig, 'Style', 'pushbutton', 'Units', 'normalized', 'Position', [0.92 0.93 0.07 0.05], ...
        'String', 'Print', 'FontName', fontName, 'FontSize', 12, 'Callback', @onPrint);

    onSliderChange();
    updateModelPlots();  % initial model update
    uiwait(fig);

    function onSliderChange(~,~)
        if ~ishandle(fig)
            return;
        end

        % ---- n bounds: scale default half-width, but do not allow below ±1% of nominal ----
        nScale = get(sNBound, 'Value');

        nCenter = nInit;
        nHalf0  = 0.5 .* (n_ub0 - n_lb0);
        nHalf   = nHalf0 .* nScale;

        nMinHalf = 0.01 .* max(nCenter, 0);    % ±1% of nominal
        nHalf    = max(nHalf, nMinHalf);

        n_lb = max(0, nCenter - nHalf);
        n_ub = max(0, nCenter + nHalf);

        % ---- k bounds: set absolute max down to 0.001 ----
        kBoundMaxAbs = get(sKBoundAbs, 'Value');
        k_lb = zeros(size(k_ub0));
        k_ub = k_ub0;

        kMax0 = max(k_ub0);
        if ~isfinite(kMax0) || kMax0 <= 0
            kMax0 = 1;
        end
        scaleK = kBoundMaxAbs / kMax0;
        k_ub = max(0, k_ub0 .* scaleK);

        set(pN, 'YData', [n_lb; flipud(n_ub)]);
        set(pK, 'YData', [k_lb; flipud(k_ub)]);

        % ---- n y-limit: centered around nominal nRef ----
        pNY = get(sNY, 'Value');
        ylim(axN, [nRef*(1 - pNY), nRef*(1 + pNY)]);

        % ---- k y-limit: min=0, max down to 0.01 ----
        kY = get(sKYAbs, 'Value');
        ylim(axK, [0, max(kY, 0.01)]);

        drawnow;
    end

    function onRecalculate(~,~)
        if ~ishandle(fig)
            return;
        end

        set(fig, 'Pointer', 'watch');
        drawnow;

        nScale = get(sNBound, 'Value');
        kBoundMaxAbs = get(sKBoundAbs, 'Value');

        [epsNew, tanNew, n_lb_new, n_ub_new, k_ub_new] = runOptimizationOnce(nScale, kBoundMaxAbs);

        epsOut = epsNew(:);
        tanOut = tanNew(:);

        % Update n/k lines
        nExt = sqrt(max(epsOut, 0));
        kExt = 0.5 .* nExt .* abs(tanOut);

        set(lnNext, 'YData', nExt);
        set(lnKext, 'YData', kExt);

        % Update bound fills with the *current* slider-scaled bounds preview
        set(pN, 'YData', [n_lb_new; flipud(n_ub_new)]);
        set(pK, 'YData', [zeros(size(k_ub_new)); flipud(k_ub_new)]);

        updateModelPlots();

        set(fig, 'Pointer', 'arrow');
        drawnow;
    end

    function [epsNew, tanNew, n_lb_plot, n_ub_plot, k_ub_plot] = runOptimizationOnce(nScale, kBoundMaxAbs)
        numPoints = numel(freq);

        epsNew = epsOut(:);
        tanNew = tanOut(:);

        n_lb_plot = zeros(numPoints,1);
        n_ub_plot = zeros(numPoints,1);
        k_ub_plot = zeros(numPoints,1);

        if strcmpi(algoType, 'GRL')
            flagA = 0;
        else
            flagA = 1;
        end

        tanScaleFactor = 1;
        if isfield(config, 'Optimization') && isfield(config.Optimization, 'SpecialBandBounds') && ...
           isfield(config.Optimization.SpecialBandBounds, 'TanScaleFactor')
            tanScaleFactor = config.Optimization.SpecialBandBounds.TanScaleFactor;
        end

        for kk = 1:numPoints
            eps0 = epsInit(kk);
            tan0 = max(tanInit(kk), 1e-5);

            % -------- Base bounds (same as before) --------
            if ~isSpecialBand
                lb_eps0 = eps0 * (1 - bf);
                ub_eps0 = eps0 * (1 + bf);

                lb_tan0 = -tanScaleFactor * tan0;
                ub_tan0 =  tanScaleFactor * tan0;
            else
                lb_eps0 = eps0 * (1 - 4*bf);
                ub_eps0 = eps0 * (1 + 4*bf);

                lb_tan0 = -1;
                ub_tan0 =  1;
            end

            % -------- Apply user n-bounds scaling (in n-space) --------
            n0     = sqrt(max(eps0, 0));
            n_lb0p = sqrt(max(lb_eps0, 0));
            n_ub0p = sqrt(max(ub_eps0, 0));

            nHalf0 = 0.5 * (n_ub0p - n_lb0p);
            nHalf  = nHalf0 * nScale;

            nMinHalf = 0.01 * max(n0, 0);
            nHalf    = max(nHalf, nMinHalf);

            n_lb = max(0, n0 - nHalf);
            n_ub = max(0, n0 + nHalf);

            lb_eps = max(0, n_lb^2);
            ub_eps = max(0, n_ub^2);

            % -------- Apply user k-bounds (absolute max down to 0.001) --------
            % Map kBoundMaxAbs -> tan bounds using k ~= 0.5*n*tan
            nSafe = max(n0, 1e-6);
            tanMaxUser = 2 * kBoundMaxAbs / nSafe;

            % Keep feasibility: must include initial tan0 at least
            tanMax = max([tanMaxUser, abs(tan0)*1.001, 1e-5]);

            lb_tan = -tanMax;
            ub_tan =  tanMax;

            % Store plotting bounds in n/k
            n_lb_plot(kk) = n_lb;
            n_ub_plot(kk) = n_ub;
            k_ub_plot(kk) = max(0, 0.5 * n_ub * tanMax);

            lb = [lb_eps; lb_tan];
            ub2 = [ub_eps; ub_tan];

            % Objective
            if strcmpi(algoType, 'GRL')
                if isstruct(S_obj)
                    S_meas_k = S_obj.S21(kk);
                else
                    S_meas_k = S_obj(kk);
                end
                objective = @(x) sum(optimizationErrorGRLalg( ...
                    x, omega(kk), thickness, S_meas_k, flagA, sigma, c0).^2 );
            else
                objective = @(x) sum(optimizationErrorTRLalg( ...
                    x, omega(kk), thickness, ...
                    S_obj.S11(kk), S_obj.S12(kk), ...
                    S_obj.S21(kk), S_obj.S22(kk), ...
                    flagA, sigma, c0).^2 );
            end

            % Use current extracted as start (clipped into bounds)
            x0 = [epsNew(kk), tanNew(kk)];
            x0(1) = min(max(x0(1), lb(1)), ub2(1));
            x0(2) = min(max(x0(2), lb(2)), ub2(2));

            try
                x = fmincon(objective, x0, [], [], [], [], lb, ub2, [], optionsFmincon);
                epsNew(kk) = x(1);
                tanNew(kk) = x(2);
            catch
                % If optimizer fails at a point, keep previous value
            end
        end
    end

    function updateModelPlots()
        if ~ishandle(fig)
            return;
        end

        if strcmpi(algoType, 'TRL') && isTRL
            flagA = sigma > 0;

            SmodelStruct = arrayfun(@(kk) modelSparams(epsOut(kk), tanOut(kk), omega(kk), thickness, flagA, sigma, c0), 1:numel(freq));
            S11_model = [SmodelStruct.S11].';
            S21_model = [SmodelStruct.S21].';
            S12_model = [SmodelStruct.S12].';
            S22_model = [SmodelStruct.S22].';

            set(lnMod(1), 'YData', 20*log10(abs(S11_model)));
            set(lnMod(2), 'YData', 20*log10(abs(S21_model)));
            set(lnMod(3), 'YData', 20*log10(abs(S12_model)));
            set(lnMod(4), 'YData', 20*log10(abs(S22_model)));

        else
            % GRL combined comparison (use the S11 slot)
            epsComplex = epsOut(:) - 1i .* tanOut(:) .* epsOut(:);
            sqrtEps    = sqrt(epsComplex);

            S_model_combined = epsComplex ./ (epsComplex .* cos(omega .* sqrtEps .* thickness ./ c0) + ...
                1i .* sqrtEps .* 0.5 .* (1 + epsComplex) .* sin(omega .* sqrtEps .* thickness ./ c0));

            set(lnMod(1), 'YData', 20*log10(abs(S_model_combined)));
        end

        drawnow;
    end

    function onSave(~,~)
        if ~ishandle(fig)
            return;
        end
        uiresume(fig);
        delete(fig);
    end

    function onPrint(~,~)
        if ~ishandle(fig)
            return;
        end

        fig_handle = fig;

        % --- A. BASIC STYLE FIXES (Black axes/text, dark gray grid) ---
        set(fig_handle, 'Color', 'w');
        all_axes_local = findall(fig_handle, 'type', 'axes');

        set(all_axes_local, 'Color', 'w');
        set(all_axes_local, 'XColor', 'k', 'YColor', 'k', 'ZColor', 'k');

        set(findall(fig_handle, 'type', 'text'), 'Color', 'k');

        set(all_axes_local, 'GridColor', [0.2 0.2 0.2]);
        set(all_axes_local, 'GridAlpha', 0.4);

        set(findall(fig_handle, 'type', 'legend'), 'Color', 'w', 'TextColor', 'k', 'EdgeColor', 'k', 'Box', 'on');

        % --- B. BOUNDS FILLS (Patch) -> Light blue ---
        dolgular = findall(fig_handle, 'Type', 'Patch');
        if ~isempty(dolgular)
            set(dolgular, 'FaceColor', [0.6 0.85 1.0]);
            set(dolgular, 'FaceAlpha', 0.3);
            set(dolgular, 'EdgeColor', 'none');
        else
            disp('Warning: No filled bound areas (Patch) found in the figure.');
        end

        % --- C. SAVE ---
        outFolder = pwd;
        baseName  = 'VNA_plot';

        if ~isempty(fileLabel)
            try
                [outFolderTmp, baseNameTmp, ~] = fileparts(char(fileLabel));
                if ~isempty(outFolderTmp), outFolder = outFolderTmp; end
                if ~isempty(baseNameTmp),  baseName  = baseNameTmp;  end
            catch
            end
        end

        baseName = regexprep(baseName, '\s+', '_');
        dosyaAdi = fullfile(outFolder, [baseName '_' algoType '_full.pdf']);

        exportgraphics(fig_handle, dosyaAdi, 'ContentType', 'vector', 'BackgroundColor', 'white');
        fprintf('Done! Saved to: %s\n', dosyaAdi);
    end

    function onClose(~,~)
        if ishandle(fig)
            uiresume(fig);
            delete(fig);
        end
    end
end

function [epsInit, tanDeltaInit] = initialGuessFromS21(freq, S21_complex, thickness, c0)
% Robust initial guess for low-loss dielectrics from S21.
% - Uses phase (group delay) for a single effective epsilon.
% - Uses magnitude for a single effective tan(delta), using a noise-robust approach.

    freq = freq(:);
    S21_complex = S21_complex(:);

    N = numel(freq);
    if N < 5
        error('Not enough frequency points for initialGuessFromS21.');
    end

    %% -------- 1) EPSILON from (smoothed) group delay --------
    phi = unwrap(angle(S21_complex));   % [rad]

    % Smooth phase noise (especially important for low-loss materials)
    if N > 21
        phi_s = smoothdata(phi, 'movmedian', 21);
    else
        phi_s = phi;
    end

    % Fit phase slope using the mid-band region
    i1 = max(1, round(0.2*N));
    i2 = min(N, round(0.8*N));
    idxMid = i1:i2;

    p = polyfit(freq(idxMid), phi_s(idxMid), 1);      % phi ≈ p(1)*f + p(2)
    slope = p(1);                                     % dphi/df [rad/Hz]

    % Group delay: tau_g = -1/(2*pi) * dphi/df
    tau_group = -slope / (2*pi);          % [s]

    % Effective index: n_eff = c0 * tau_g / L
    n_eff = (c0 * tau_group) / thickness;

    % Clamp n_eff to a physical range
    n_eff = max(n_eff, 1.02);   % lower limit
    n_eff = min(n_eff, 6.0);    % upper limit

    eps_scalar = n_eff^2;
    epsInit = eps_scalar * ones(size(freq));

    %% -------- 2) TANDELTA from magnitude (noise-robust) --------
    magS21 = abs(S21_complex);
    magS21(magS21 < 1e-9) = 1e-9;    % avoid log(0)

    sqrt_eps = sqrt(eps_scalar);
    Gamma    = (sqrt_eps - 1) / (sqrt_eps + 1);
    T        = 1 - Gamma^2;         % transmission factor without loss

    % |S21| ≈ T * exp(-alpha * L)  => alpha = -(1/L) * ln(|S21| / T)
    alpha = -(1/thickness) * log(magS21 ./ T);
    alpha(~isfinite(alpha)) = 0;
    alpha(alpha < 0) = 0;

    % Use mid–high band region (more robust)
    idxAlphaBand = (freq >= freq(round(0.3*N))) & (freq <= freq(round(0.9*N)));
    alphaPos = alpha(idxAlphaBand & alpha > 0);

    if isempty(alphaPos)
        alpha_med = 0;
    else
        alpha_med = median(alphaPos);
    end

    % For dielectrics: alpha ≈ (beta * tanδ) / 2
    omega = 2*pi*freq;
    beta = (omega .* sqrt_eps) ./ c0;

    beta_mid = mean(beta(idxAlphaBand));

    if beta_mid <= 0 || ~isfinite(beta_mid)
        tanDelta_scalar = 0.01;
    else
        tanDelta_scalar = 2 * alpha_med / beta_mid;
    end

    tanDeltaInit = tanDelta_scalar * ones(size(freq));
end
