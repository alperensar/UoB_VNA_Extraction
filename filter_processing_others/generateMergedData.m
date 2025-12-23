function generateMergedData()
% generateMergedData
%
% Batch processing tool to MERGE + AVERAGE data inside each band folder.
%
% Output location (per band):
%   <bandFolder>/<MergedSubFolder>/
%
% OPTIONAL: "MergedBands check"
% - If enabled, the function scans <methodFolder>/<MergedBandsName>/ for combo folders
%   (e.g., WR15_WR10) and if combo outputs exist it can skip processing those member bands
%   for that data type, and also copy combo outputs into:
%       <methodFolder>/<MergedBandsName>/<MergedSubFolder>/
%
% Rules:
% - If a combo folder name contains a band name (e.g., WR15_WR10 contains WR15):
%   • If perm outputs exist for that combo -> skip PERMITTIVITY for WR15 (and WR10)
%   • If s2p outputs exist for that combo  -> skip S2P for WR15 (and WR10)
% - Band outputs remain in the original place:
%     <bandFolder>/<MergedSubFolder>/
% - Do NOT create <bandFolder>/<MergedSubFolder> unless something is processed.
%
% JSON requirements (config_VNA.json):
%   config.MergedProcessing.Output.MergedSubFolder
%   config.MergedProcessing.Output.PermittivityMergedSuffix
%   config.MergedProcessing.Output.PermittivityAveragedSuffix
%   config.MergedProcessing.Output.S2PMergedSuffix
%   config.MergedProcessing.Output.S2PAveragedSuffix   (NEW for S2P averaged)
%
% Permittivity:
% - Merged: columns for each file appended side-by-side (Freq/Eps/Tan/Thk).
% - Averaged: mean + std per frequency (epsReal, tanDelta), thickness passed through.
%
% S2P:
% - Merged: columns for each file appended side-by-side (Freq + 8 S-parameter columns).
% - Averaged: complex-mean per frequency for each Sij (output as mag + phase).
%
% Notes:
% - S2P averaging uses complex mean: mag*exp(j*phase).
% - Column labels follow the same "k_Header" convention as before.

    % ---------------- CONFIG ----------------
    configPath = 'config_VNA.json';
    if ~isfile(configPath)
        error('Could not find config file: %s', configPath);
    end
    configText = fileread(configPath);
    config = jsondecode(configText);

    if ~isfield(config, "MergedProcessing")
        error('config.MergedProcessing block is not defined in JSON.');
    end
    mp = config.MergedProcessing;

    if ~isfield(mp, "Output")
        error('config.MergedProcessing.Output is missing in JSON.');
    end

    % Output suffixes/folders (ensure char for pattern concatenations)
    mergedSubFolder     = char(mp.Output.MergedSubFolder);
    permMergedSuffix    = char(mp.Output.PermittivityMergedSuffix);
    permAveragedSuffix  = char(mp.Output.PermittivityAveragedSuffix);
    s2pMergedSuffix     = char(mp.Output.S2PMergedSuffix);

    if isfield(mp.Output, "S2PAveragedSuffix") && ~isempty(mp.Output.S2PAveragedSuffix)
        s2pAveragedSuffix = char(mp.Output.S2PAveragedSuffix);
    else
        s2pAveragedSuffix = '_S2PAveraged_Data.csv';
    end

    % Output headers
    if isfield(mp.Output, "PermittivityMergedHeaders")
        permMergedHeaders = mp.Output.PermittivityMergedHeaders;
    else
        permMergedHeaders = {'Freq_Hz', 'EpsReal', 'TanDelta', 'Thickness_mm'};
    end

    if isfield(mp.Output, "PermittivityAveragedHeaders")
        permAvgHeaders = mp.Output.PermittivityAveragedHeaders;
    else
        permAvgHeaders = {'Freq_Hz', 'EpsReal_Mean', 'EpsReal_STD', 'TanDelta_Mean', 'TanDelta_STD', 'Thickness_mm'};
    end

    if isfield(mp.Output, "S2PHeaders")
        s2pHeaders = mp.Output.S2PHeaders;
    else
        s2pHeaders = {'Freq_Hz', 'S11_mag', 'S11_phase', 'S21_mag', 'S21_phase', 'S12_mag', 'S12_phase', 'S22_mag', 'S22_phase'};
    end

    if isfield(mp.Output, "S2PAveragedHeaders")
        s2pAvgHeaders = mp.Output.S2PAveragedHeaders;
    else
        s2pAvgHeaders = {'Freq_Hz', ...
            'S11_mag_Mean','S11_phase_Mean', ...
            'S21_mag_Mean','S21_phase_Mean', ...
            'S12_mag_Mean','S12_phase_Mean', ...
            'S22_mag_Mean','S22_phase_Mean'};
    end

    % Show data format info from JSON
    infoLines = message2screen(mp, s2pAveragedSuffix);
    msg = strjoin(infoLines, newline);
    userChoice = questdlg(msg, 'Merged Data Format Info', 'Continue', 'Cancel', 'Cancel');
    if ~strcmp(userChoice, 'Continue')
        disp('Operation cancelled by user.');
        return;
    end

    % Root folder selection either from JSON or interactive
    useConfigDir = false;
    if isfield(config, "FilePatterns") && isfield(config.FilePatterns, 'rootDir') && ~isempty(config.FilePatterns.rootDir)
        if isfolder(config.FilePatterns.rootDir)
            useConfigDir = true;
        end
    end

    if useConfigDir
        rootDir = config.FilePatterns.rootDir;
        if isstring(rootDir), rootDir = char(rootDir); end
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

    % ---------------- OPTIONAL: MergedBands check ----------------
    mergedBandsCheck = false;
    mergedBandsName  = '';

    btn = questdlg( ...
        "Enable 'MergedBands check'? (Skip band types if combo outputs exist + copy combo outputs into MergedBands/<MergedSubFolder>/)", ...
        "MergedBands Check", "No", "Yes", "No");

    if strcmp(btn, "Yes")
        answ = inputdlg("Enter folder name (e.g., MergedBands):", "MergedBands Folder", 1, {"MergedBands"});
        if isempty(answ) || isempty(strtrim(answ{1}))
            disp('Operation cancelled by user.');
            return;
        end
        mergedBandsCheck = true;
        mergedBandsName  = strtrim(answ{1});
    end

    % All sample folders
    sampleFolders = dir(rootDir);
    sampleFolders = sampleFolders([sampleFolders.isdir] & ~startsWith({sampleFolders.name}, '.'));

    % Main loops: sample → method → band
    for s = 1:numel(sampleFolders)
        sampleFolderPath = fullfile(rootDir, sampleFolders(s).name);

        for m = 1:numel(mp.Methods)
            method = mp.Methods(m);
            methodFolderPath = fullfile(sampleFolderPath, method.FolderName);
            if ~exist(methodFolderPath, 'dir')
                continue;
            end

            bands = method.Bands;

            if isstruct(bands)
                bandNamesJSON = {bands.Name};
            elseif iscell(bands)
                bandNamesJSON = cellfun(@(x) x.Name, bands, 'UniformOutput', false);
            elseif isstring(bands)
                bandNamesJSON = cellstr(bands);
            elseif ischar(bands)
                bandNamesJSON = {bands};
            else
                error("Unsupported type for method.Bands: %s", class(bands));
            end

            % Per-method pre-scan (coverage lists)
            permCoveredBands = strings(0,1);
            s2pCoveredBands  = strings(0,1);

            if mergedBandsCheck
                mergedBandsPath = fullfile(methodFolderPath, mergedBandsName);
                if isfolder(mergedBandsPath)
                    [permCoveredBands, s2pCoveredBands] = mergedBands_collectAndDetect_mb( ...
                        mergedBandsPath, bandNamesJSON, mergedSubFolder, ...
                        permMergedSuffix, permAveragedSuffix, s2pMergedSuffix, s2pAveragedSuffix);
                end
            end

            for b = 1:numel(bands)
                bandName = string(bands(b).Name);

                lowerCut = -inf;
                upperCut =  inf;

                if isfield(bands(b),"LowerCutoffHz")
                    lowerCut = bands(b).LowerCutoffHz;
                end
                if isfield(bands(b),"UpperCutoffHz")
                    upperCut = bands(b).UpperCutoffHz;
                end

                bandFolderPath = fullfile(methodFolderPath, char(bandName));
                if ~exist(bandFolderPath, 'dir')
                    continue;
                end

                % Decide skip per type ONLY by MergedBands combo outputs
                skipPerm = mergedBandsCheck && any(strcmpi(bandName, permCoveredBands));
                skipS2P  = mergedBandsCheck && any(strcmpi(bandName, s2pCoveredBands));

                % All files in this band
                inputFiles = dir(fullfile(bandFolderPath, '*'));
                inputFiles = inputFiles(~[inputFiles.isdir]);
                if isempty(inputFiles)
                    continue;
                end

                [~, ~, exts] = cellfun(@fileparts, {inputFiles.name}, 'UniformOutput', false);
                exts = lower(exts);

                % Determine if there is any input for each type
                hasPerm = false;
                hasS2P  = false;

                if isfield(method, "PermittivityInput")
                    permCfg  = method.PermittivityInput;
                    permExts = lower(string(permCfg.Extensions));
                    hasPerm  = any(ismember(exts, permExts));
                end

                if isfield(method, "SParameterInput")
                    sp = method.SParameterInput;
                    s2pExts = lower(string(sp.Extensions));
                    hasS2P  = any(ismember(exts, s2pExts));
                end

                doPerm = ~skipPerm && isfield(method, "PermittivityInput") && hasPerm;
                doS2P  = ~skipS2P  && isfield(method, "SParameterInput")   && hasS2P;

                % If everything is skipped (or no inputs), do NOT create empty Merged folder
                if ~doPerm && ~doS2P
                    continue;
                end

                % Output folder (ONLY create when needed)
                outputFolder = fullfile(bandFolderPath, mergedSubFolder);
                if ~exist(outputFolder, 'dir')
                    mkdir(outputFolder);
                end

                % -------------------- PERMITTIVITY (CSV/TXT) --------------------
                if doPerm
                    permCfg   = method.PermittivityInput;
                    permExts  = lower(string(permCfg.Extensions));
                    isPerm    = ismember(exts, permExts);
                    permFiles = inputFiles(isPerm);

                    if ~isempty(permFiles)
                        mergedDataMatrix = [];
                        bandDataList = cell(numel(permFiles), 1);

                        for f = 1:numel(permFiles)
                            try
                                filePath = fullfile(bandFolderPath, permFiles(f).name);
                                numericData = parsePermittivityFile(filePath, permCfg, lowerCut, upperCut);
                                if isempty(numericData)
                                    continue;
                                end

                                bandDataList{f} = numericData;

                                % Pad rows & append horizontally
                                nRows = size(numericData,1);
                                existingRows = size(mergedDataMatrix,1);
                                maxRows = max(nRows, existingRows);

                                if existingRows < maxRows && ~isempty(mergedDataMatrix)
                                    mergedDataMatrix(end+1:maxRows,:) = NaN;
                                end
                                if nRows < maxRows && ~isempty(mergedDataMatrix)
                                    numericData(end+1:maxRows,:) = NaN;
                                end

                                mergedDataMatrix = [mergedDataMatrix numericData]; %#ok<AGROW>

                            catch ME
                                fprintf('Warning: error processing permittivity file %s: %s\n', permFiles(f).name, ME.message);
                            end
                        end

                        bandDataList = bandDataList(~cellfun(@isempty, bandDataList));

                        if ~isempty(mergedDataMatrix) && ~isempty(bandDataList)
                            % Averaging & STD over all files for each frequency
                            allData = vertcat(bandDataList{:});
                            [uniqueFreqs, ~, freqGroup] = unique(allData(:,1));
                            dataCols = allData(:,2:end);

                            meanVals = zeros(numel(uniqueFreqs), size(dataCols,2));
                            stdVals  = zeros(numel(uniqueFreqs), size(dataCols,2));

                            for u = 1:numel(uniqueFreqs)
                                rows = dataCols(freqGroup == u, :);

                                % Example sanity: ignore negative epsilon rows
                                rows(rows(:,2) < 0, 1:3) = NaN;

                                meanVals(u,:) = mean(rows, 1, 'omitnan');
                                stdVals(u,:)  = std(rows, 0, 1, 'omitnan');
                            end

                            averagedData = [uniqueFreqs, meanVals(:,1), stdVals(:,1), meanVals(:,2), stdVals(:,2), meanVals(:,3)];

                            % Build output base name from band path
                            relativeBandPath = strrep(bandFolderPath, [rootDir filesep], '');
                            outputBaseName = strjoin(strsplit(relativeBandPath, filesep), '_');

                            % Save averaged CSV
                            writetable(array2table(averagedData, 'VariableNames', permAvgHeaders), ...
                                fullfile(outputFolder, [outputBaseName permAveragedSuffix]));

                            % Save merged CSV
                            colLabels = strings(1, numel(permFiles) * numel(permMergedHeaders));
                            idx = 1;
                            for ff = 1:numel(permFiles)
                                for h = 1:numel(permMergedHeaders)
                                    colLabels(idx) = sprintf('%d_%s', ff, permMergedHeaders{h});
                                    idx = idx + 1;
                                end
                            end

                            writetable(array2table(mergedDataMatrix, 'VariableNames', cellstr(colLabels)), ...
                                fullfile(outputFolder, [outputBaseName permMergedSuffix]));

                            fprintf('%s: permittivity merged + averaged saved.\n', outputBaseName);
                        end
                    end
                end

                % ------------------------ S2P MERGE + AVERAGE ------------------------
                if doS2P
                    sp = method.SParameterInput;
                    s2pExts  = lower(string(sp.Extensions));
                    isS2P    = ismember(exts, s2pExts);
                    s2pFiles = inputFiles(isS2P);

                    if ~isempty(s2pFiles)
                        % Determine file order from last number in filename (optional)
                        if isfield(sp, "UseFilenameLastNumberAsOrder") && sp.UseFilenameLastNumberAsOrder
                            s2pOrder = zeros(numel(s2pFiles),1);
                            for f = 1:numel(s2pFiles)
                                [~, nameNoExt, ~] = fileparts(s2pFiles(f).name);
                                numStr = regexp(nameNoExt, '(\d+)(?!.*\d)', 'match', 'once');
                                if ~isempty(numStr)
                                    s2pOrder(f) = str2double(numStr);
                                else
                                    s2pOrder(f) = f;
                                end
                            end
                            [~, sortIdx] = sort(s2pOrder);
                            s2pFiles = s2pFiles(sortIdx);
                        end

                        s2pMergedMatrix = [];
                        s2pDataList = cell(numel(s2pFiles), 1);

                        for f = 1:numel(s2pFiles)
                            try
                                filePath = fullfile(bandFolderPath, s2pFiles(f).name);
                                numericS2P = readS2PAsMatrix(filePath, sp.Columns, sp);

                                % Optional band cutoff
                                if isfield(sp, "ApplyBandCutoff") && sp.ApplyBandCutoff
                                    numericS2P = numericS2P(numericS2P(:,1) >= lowerCut & numericS2P(:,1) <= upperCut, :);
                                end

                                if isempty(numericS2P)
                                    continue;
                                end

                                s2pDataList{f} = numericS2P;

                                % Pad & append for merged matrix
                                nRows = size(numericS2P,1);
                                existingRows = size(s2pMergedMatrix,1);
                                maxRows = max(nRows, existingRows);

                                if existingRows < maxRows && ~isempty(s2pMergedMatrix)
                                    s2pMergedMatrix(end+1:maxRows,:) = NaN;
                                end
                                if nRows < maxRows && ~isempty(s2pMergedMatrix)
                                    numericS2P(end+1:maxRows,:) = NaN;
                                end

                                s2pMergedMatrix = [s2pMergedMatrix numericS2P]; %#ok<AGROW>

                            catch ME
                                fprintf('Warning: error processing S2P file %s: %s\n', s2pFiles(f).name, ME.message);
                            end
                        end

                        s2pDataList = s2pDataList(~cellfun(@isempty, s2pDataList));

                        % Build output base name from band path
                        relativeBandPath = strrep(bandFolderPath, [rootDir filesep], '');
                        outputBaseName   = strjoin(strsplit(relativeBandPath, filesep), '_');

                        % ---- Save merged S2P ----
                        if ~isempty(s2pMergedMatrix)
                            colLabelsS2P = strings(1, numel(s2pFiles) * numel(s2pHeaders));
                            idx = 1;
                            for ff = 1:numel(s2pFiles)
                                for h = 1:numel(s2pHeaders)
                                    colLabelsS2P(idx) = sprintf('%d_%s', ff, s2pHeaders{h});
                                    idx = idx + 1;
                                end
                            end

                            outS2PFile = fullfile(outputFolder, [outputBaseName s2pMergedSuffix]);
                            writetable(array2table(s2pMergedMatrix, 'VariableNames', cellstr(colLabelsS2P)), outS2PFile);
                            fprintf('%s: S2P merged saved: %s\n', outputBaseName, outS2PFile);
                        end

                        % ---- Save averaged S2P ----
                        if ~isempty(s2pDataList)
                            allS = vertcat(s2pDataList{:}); % [f, mag/phase ...]
                            [uniqueFreqs, ~, g] = unique(allS(:,1));

                            avgS = NaN(numel(uniqueFreqs), 9);
                            avgS(:,1) = uniqueFreqs;

                            for u = 1:numel(uniqueFreqs)
                                rows = allS(g==u, :);

                                % Complex mean for each Sij (mag*exp(j*phase))
                                S11 = rows(:,2) .* exp(1j*deg2rad(rows(:,3)));
                                S21 = rows(:,4) .* exp(1j*deg2rad(rows(:,5)));
                                S12 = rows(:,6) .* exp(1j*deg2rad(rows(:,7)));
                                S22 = rows(:,8) .* exp(1j*deg2rad(rows(:,9)));

                                mS11 = mean(S11, 'omitnan');
                                mS21 = mean(S21, 'omitnan');
                                mS12 = mean(S12, 'omitnan');
                                mS22 = mean(S22, 'omitnan');

                                avgS(u,2) = abs(mS11);
                                avgS(u,3) = rad2deg(angle(mS11));
                                avgS(u,4) = abs(mS21);
                                avgS(u,5) = rad2deg(angle(mS21));
                                avgS(u,6) = abs(mS12);
                                avgS(u,7) = rad2deg(angle(mS12));
                                avgS(u,8) = abs(mS22);
                                avgS(u,9) = rad2deg(angle(mS22));
                            end

                            outS2PAvgFile = fullfile(outputFolder, [outputBaseName s2pAveragedSuffix]);
                            writetable(array2table(avgS, 'VariableNames', s2pAvgHeaders), outS2PAvgFile);
                            fprintf('%s: S2P averaged saved: %s\n', outputBaseName, outS2PAvgFile);
                        end
                    end
                end
            end
        end
    end

    fprintf('\nAll samples (selected method folders) processed successfully.\n');
end


function [permCoveredBands, s2pCoveredBands] = mergedBands_collectAndDetect_mb( ...
    mergedBandsPath, bandNamesJSON, mergedSubFolder, ...
    permMergedSuffix, permAveragedSuffix, s2pMergedSuffix, s2pAveragedSuffix)

    % Central destination: <MergedBands>/<MergedSubFolder>/
    centralMerged = fullfile(mergedBandsPath, mergedSubFolder);
    if ~exist(centralMerged, 'dir')
        mkdir(centralMerged);
    end

    permCoveredBands = strings(0,1);
    s2pCoveredBands  = strings(0,1);

    combos = dir(mergedBandsPath);
    combos = combos([combos.isdir] & ~startsWith({combos.name}, '.'));

    % Do not treat the mergedSubFolder itself as a combo folder
    combos = combos(~strcmpi(string({combos.name}), string(mergedSubFolder)));

    for c = 1:numel(combos)
        comboName = string(combos(c).name);
        comboPath = fullfile(mergedBandsPath, char(comboName));

        members = parseComboMembers_mb(comboName, bandNamesJSON);
        if isempty(members)
            continue;
        end

        % Look for outputs in:
        % 1) combo/<MergedSubFolder>/
        % 2) combo/   (fallback)
        src1 = fullfile(comboPath, mergedSubFolder);
        src2 = comboPath;

        permFiles = [];
        s2pFiles  = [];

        permFiles = [permFiles; dir(fullfile(src1, ['*' permMergedSuffix]))]; %#ok<AGROW>
        permFiles = [permFiles; dir(fullfile(src1, ['*' permAveragedSuffix]))]; %#ok<AGROW>
        permFiles = [permFiles; dir(fullfile(src2, ['*' permMergedSuffix]))]; %#ok<AGROW>
        permFiles = [permFiles; dir(fullfile(src2, ['*' permAveragedSuffix]))]; %#ok<AGROW>

        s2pFiles  = [s2pFiles;  dir(fullfile(src1, ['*' s2pMergedSuffix]))]; %#ok<AGROW>
        s2pFiles  = [s2pFiles;  dir(fullfile(src1, ['*' s2pAveragedSuffix]))]; %#ok<AGROW>
        s2pFiles  = [s2pFiles;  dir(fullfile(src2, ['*' s2pMergedSuffix]))]; %#ok<AGROW>
        s2pFiles  = [s2pFiles;  dir(fullfile(src2, ['*' s2pAveragedSuffix]))]; %#ok<AGROW>

        % Copy discovered outputs into CENTRAL folder (no overwrite)
        copyFilesIfMissing_mb(permFiles, centralMerged);
        copyFilesIfMissing_mb(s2pFiles,  centralMerged);

        % Consider it "done" if files exist either found OR already in central folder.
        permDone = ~isempty(permFiles) || ...
                   ~isempty(dir(fullfile(centralMerged, ['*' char(comboName) '*' permMergedSuffix]))) || ...
                   ~isempty(dir(fullfile(centralMerged, ['*' char(comboName) '*' permAveragedSuffix])));

        s2pDone  = ~isempty(s2pFiles)  || ...
                   ~isempty(dir(fullfile(centralMerged, ['*' char(comboName) '*' s2pMergedSuffix]))) || ...
                   ~isempty(dir(fullfile(centralMerged, ['*' char(comboName) '*' s2pAveragedSuffix])));

        if permDone
            permCoveredBands = unique([permCoveredBands; members(:)]);
        end
        if s2pDone
            s2pCoveredBands = unique([s2pCoveredBands; members(:)]);
        end
    end
end

function members = parseComboMembers_mb(comboName, bandNamesJSON)
    % Preferred: split by '_' and exact token match
    toks = strsplit(string(comboName), "_");
    members = strings(0,1);

    for i = 1:numel(toks)
        t = string(toks{i});
        hit = bandNamesJSON(strcmpi(bandNamesJSON, t));
        if ~isempty(hit)
            members(end+1,1) = hit(1); %#ok<AGROW>
        end
    end

    % Fallback: contains-based match
    if isempty(members)
        nm = lower(string(comboName));
        for i = 1:numel(bandNamesJSON)
            bn = lower(string(bandNamesJSON(i)));
            if contains(nm, bn)
                members(end+1,1) = bandNamesJSON(i); %#ok<AGROW>
            end
        end
        members = unique(members);
    end
end

function copyFilesIfMissing_mb(fileList, dstFolder)
    if isempty(fileList)
        return;
    end

    if ~exist(dstFolder, 'dir')
        mkdir(dstFolder);
    end

    for k = 1:numel(fileList)
        src = fullfile(fileList(k).folder, fileList(k).name);
        dst = fullfile(dstFolder, fileList(k).name);

        if ~isfile(dst)
            try
                copyfile(src, dst);
            catch
                % ignore copy errors
            end
        end
    end
end


function numericData = parsePermittivityFile(filePath, permCfg, lowerCut, upperCut)
    % Reads a permittivity-type file (CSV/TXT) according to permCfg:
    % Supports band cutoffs:
    %   freq >= lowerCut  AND  freq <= upperCut
    %
    % Output:
    %   [Freq_Hz, EpsReal, TanDelta, Thickness_mm_marker]
    % Thickness marker is only placed in row 1 (like the original code).

    fileContents = readcell(filePath);

    % Remove header rows
    skip = 0;
    if isfield(permCfg, "HeaderRowsToSkip")
        skip = permCfg.HeaderRowsToSkip;
    end
    dataBlock = fileContents(skip + 1:end, :);

    % Select frequency / epsilon-real / third column
    cols = [permCfg.FrequencyColumn, permCfg.EpsRealColumn, permCfg.TanColumn];
    rawBlock = dataBlock(:, cols);

    % Robust conversion to double
    numeric = nan(size(rawBlock,1),3);
    for r = 1:size(rawBlock,1)
        for c = 1:3
            v = rawBlock{r,c};
            if isnumeric(v)
                numeric(r,c) = double(v);
            else
                numeric(r,c) = str2double(string(v));
            end
        end
    end
    numeric = rmmissing(numeric);
    if isempty(numeric)
        numericData = [];
        return;
    end

    freq    = numeric(:,1);
    epsReal = numeric(:,2);
    col3    = numeric(:,3);

    % Interpret TanColumn as tanδ or ε'' based on JSON
    if isfield(permCfg, "TanDelta_EpsImag")
        switch lower(string(permCfg.TanDelta_EpsImag))
            case "tandelta"
                tanDelta = col3;
            case "epsimag"
                tanDelta = col3 ./ epsReal;
                tanDelta(~isfinite(tanDelta)) = NaN;
            otherwise
                error("Unsupported TanDelta_EpsImag: %s", string(permCfg.TanDelta_EpsImag));
        end
    else
        tanDelta = col3;
    end

    % Frequency unit conversion
    if isfield(permCfg, "FreqUnit")
        u = upper(string(permCfg.FreqUnit));
        switch u
            case "HZ"
            case "KHZ", freq = freq * 1e3;
            case "MHZ", freq = freq * 1e6;
            case "GHZ", freq = freq * 1e9;
            otherwise
                error("Unsupported permittivity frequency unit: %s", u);
        end
    end

    numeric = [freq, epsReal, tanDelta];

    % Band cutoff
    if isfield(permCfg, "ApplyBandCutoff") && permCfg.ApplyBandCutoff
        numeric = numeric(numeric(:,1) >= lowerCut & numeric(:,1) <= upperCut, :);
    end
    if isempty(numeric)
        numericData = [];
        return;
    end

    % Thickness marker (only first row)
    thicknessVal = NaN;
    if isfield(permCfg, "Thickness")
        th = permCfg.Thickness;
        if isfield(th, "Mode")
            switch lower(string(th.Mode))
                case "cell"
                    rawCell = fileContents{th.Row, th.Col};
                    thickStr = string(rawCell);
                    if isfield(th, "RemoveString") && ~isempty(th.RemoveString)
                        thickStr = strrep(thickStr, string(th.RemoveString), "");
                    end
                    thicknessVal = str2double(thickStr);
                case "constant"
                    if isfield(th, "Value")
                        thicknessVal = double(th.Value);
                    end
            end
        end
    end

    thCol = NaN(size(numeric,1),1);
    if ~isnan(thicknessVal)
        thCol(1) = thicknessVal;
    end

    numericData = [numeric, thCol];
end

function numericData = readS2PAsMatrix(filePath, columnsCfg, spCfg)
    % Reads an S2P (Touchstone) file and returns:
    % [Freq_Hz, S11_mag, S11_phase_deg, S21_mag, S21_phase_deg, S12_mag, S12_phase_deg, S22_mag, S22_phase_deg]
    % Touchstone header can override default freq unit and data format.

    fid = fopen(filePath, 'r');
    if fid == -1
        error('File could not be opened: %s', filePath);
    end

    lines = {};
    headerFreqUnit = '';
    headerFormat   = '';

    defaultFormat  = "MA";
    if isfield(spCfg, "DataFormat") && ~isempty(spCfg.DataFormat)
        defaultFormat = upper(string(spCfg.DataFormat));
    end

    defaultFreqUnit = "HZ";
    if isfield(spCfg, "FreqUnit") && ~isempty(spCfg.FreqUnit)
        defaultFreqUnit = upper(string(spCfg.FreqUnit));
    end

    while ~feof(fid)
        line = strtrim(fgetl(fid));
        if ~ischar(line), break; end
        if isempty(line), continue; end

        if startsWith(line, '#')
            tokens = upper(string(strsplit(line)));

            % frequency unit
            freqTokens = ["HZ", "KHZ", "MHZ", "GHZ"];
            hit = freqTokens(ismember(freqTokens, tokens));
            if ~isempty(hit)
                headerFreqUnit = char(hit(1));
            end

            % data format
            fmtTokens = ["DB", "MA", "RI"];
            hit = fmtTokens(ismember(fmtTokens, tokens));
            if ~isempty(hit)
                headerFormat = char(hit(1));
            end
            continue;
        end

        % comments
        if startsWith(line, '!')
            continue;
        end

        lines{end+1} = line; %#ok<AGROW>
    end
    fclose(fid);

    if isempty(lines)
        numericData = [];
        return;
    end

    data = cellfun(@(l) sscanf(l, '%f')', lines, 'UniformOutput', false);
    data = vertcat(data{:});
    data = rmmissing(data);

    % effective frequency unit
    effFreqUnit = defaultFreqUnit;
    if ~isempty(headerFreqUnit)
        effFreqUnit = upper(string(headerFreqUnit));
    end

    freq = data(:, columnsCfg.Frequency);
    switch effFreqUnit
        case "HZ"
        case "KHZ", freq = freq * 1e3;
        case "MHZ", freq = freq * 1e6;
        case "GHZ", freq = freq * 1e9;
        otherwise
            error('Unsupported frequency unit: %s', effFreqUnit);
    end

    % effective format
    effFormat = defaultFormat;
    if ~isempty(headerFormat)
        effFormat = upper(string(headerFormat));
    end

    c = columnsCfg;

    switch effFormat
        case "DB"
            dB2mag = @(d) 10.^(d/20);

            S11_mag = dB2mag(data(:,c.S11_mag)); S11_ph = data(:,c.S11_phase);
            S21_mag = dB2mag(data(:,c.S21_mag)); S21_ph = data(:,c.S21_phase);
            S12_mag = dB2mag(data(:,c.S12_mag)); S12_ph = data(:,c.S12_phase);
            S22_mag = dB2mag(data(:,c.S22_mag)); S22_ph = data(:,c.S22_phase);

        case "MA"
            S11_mag = data(:,c.S11_mag); S11_ph = data(:,c.S11_phase);
            S21_mag = data(:,c.S21_mag); S21_ph = data(:,c.S21_phase);
            S12_mag = data(:,c.S12_mag); S12_ph = data(:,c.S12_phase);
            S22_mag = data(:,c.S22_mag); S22_ph = data(:,c.S22_phase);

        case "RI"
            S11_mag = hypot(data(:,c.S11_mag), data(:,c.S11_phase));
            S11_ph  = atan2d(data(:,c.S11_phase), data(:,c.S11_mag));

            S21_mag = hypot(data(:,c.S21_mag), data(:,c.S21_phase));
            S21_ph  = atan2d(data(:,c.S21_phase), data(:,c.S21_mag));

            S12_mag = hypot(data(:,c.S12_mag), data(:,c.S12_phase));
            S12_ph  = atan2d(data(:,c.S12_phase), data(:,c.S12_mag));

            S22_mag = hypot(data(:,c.S22_mag), data(:,c.S22_phase));
            S22_ph  = atan2d(data(:,c.S22_phase), data(:,c.S22_mag));

        otherwise
            error("Unsupported S2P DataFormat: %s", effFormat);
    end

    numericData = [freq, S11_mag, S11_ph, S21_mag, S21_ph, S12_mag, S12_ph, S22_mag, S22_ph];
end

function infoLines = message2screen(mp, s2pAveragedSuffixResolved)
    % Build info message from JSON (English)

    infoLines = strings(0,1);
    infoLines(end+1) = "Data format details:";

    permMergedSuffix   = string(mp.Output.PermittivityMergedSuffix);
    permAveragedSuffix = string(mp.Output.PermittivityAveragedSuffix);
    s2pMergedSuffix    = string(mp.Output.S2PMergedSuffix);
    s2pAveragedSuffix  = string(s2pAveragedSuffixResolved);

    for m = 1:numel(mp.Methods)
        method = mp.Methods(m);

        infoLines(end+1) = "";
        infoLines(end+1) = "• Method '" + string(method.Name) + ...
                           "' (folder: '" + string(method.FolderName) + "'):";

        % Permittivity input
        if isfield(method, "PermittivityInput")
            pi = method.PermittivityInput;
            extStr = strjoin(string(pi.Extensions), ", ");

            infoLines(end+1) = "   - Permittivity files (" + extStr + "):";
            infoLines(end+1) = "     • Frequency column: "      + string(pi.FrequencyColumn);
            infoLines(end+1) = "     • Epsilon (real) column: " + string(pi.EpsRealColumn);
            infoLines(end+1) = "     • Loss column: "           + string(pi.TanColumn);
            infoLines(end+1) = "     • Header rows to skip: "   + string(pi.HeaderRowsToSkip);

            if isfield(pi, "FreqUnit")
                infoLines(end+1) = "     • Frequency unit (input): " + string(pi.FreqUnit);
            end

            if isfield(pi, "TanDelta_EpsImag")
                infoLines(end+1) = "     • Loss interpretation: " + string(pi.TanDelta_EpsImag) + ...
                                   " (TanDelta or EpsImag)";
            end

            if isfield(pi, "Thickness")
                th = pi.Thickness;
                if isfield(th, "Mode")
                    switch lower(string(th.Mode))
                        case "cell"
                            removeStr = "";
                            if isfield(th, "RemoveString") && ~isempty(th.RemoveString)
                                removeStr = ", remove '" + string(th.RemoveString) + "'";
                            end
                            unitStr = "";
                            if isfield(th, "Unit") && ~isempty(th.Unit)
                                unitStr = ", unit: " + string(th.Unit);
                            end
                            infoLines(end+1) = "     • Thickness: read from cell (" + ...
                                string(th.Row) + "," + string(th.Col) + ")" + removeStr + unitStr;

                        case "constant"
                            valStr = "NaN";
                            if isfield(th, "Value")
                                valStr = string(th.Value);
                            end
                            unitStr = "";
                            if isfield(th, "Unit") && ~isempty(th.Unit)
                                unitStr = " " + string(th.Unit);
                            end
                            infoLines(end+1) = "     • Thickness: constant " + valStr + unitStr;

                        otherwise
                            infoLines(end+1) = "     • Thickness: mode '" + string(th.Mode) + "'";
                    end
                end
            end

            if isfield(pi, "ApplyBandCutoff") && pi.ApplyBandCutoff
                infoLines(end+1) = "     • Band cutoff: enabled (uses LowerCutoffHz/UpperCutoffHz per band in JSON).";
            else
                infoLines(end+1) = "     • Band cutoff: disabled.";
            end
        end

        % S-parameter input
        if isfield(method, "SParameterInput")
            sp = method.SParameterInput;
            extStrSP = strjoin(string(sp.Extensions), ", ");

            infoLines(end+1) = "   - S-parameter files (" + extStrSP + "):";

            if isfield(sp, "FreqUnit")
                infoLines(end+1) = "     • Frequency unit (default): " + string(sp.FreqUnit) + ...
                                   " (Touchstone header can override)";
            end

            if isfield(sp, "DataFormat")
                infoLines(end+1) = "     • Data format (default): " + string(sp.DataFormat) + ...
                                   " (DB / MA / RI; Touchstone header can override)";
            end

            if isfield(sp, "UseFilenameLastNumberAsOrder") && sp.UseFilenameLastNumberAsOrder
                infoLines(end+1) = "     • File order: last number in filename (e.g., '... 1.s2p' -> 1).";
            else
                infoLines(end+1) = "     • File order: folder listing order (no numeric sorting).";
            end

            if isfield(sp, "ApplyBandCutoff") && sp.ApplyBandCutoff
                infoLines(end+1) = "     • Band cutoff: enabled (uses LowerCutoffHz/UpperCutoffHz per band in JSON).";
            else
                infoLines(end+1) = "     • Band cutoff: disabled.";
            end

            if isfield(sp, "Columns")
                c = sp.Columns;
                infoLines(end+1) = "     • Input columns (1-based):";
                infoLines(end+1) = "       - Frequency: " + string(c.Frequency);
                infoLines(end+1) = "       - S11_mag: "   + string(c.S11_mag) + ", S11_phase: " + string(c.S11_phase);
                infoLines(end+1) = "       - S21_mag: "   + string(c.S21_mag) + ", S21_phase: " + string(c.S21_phase);
                infoLines(end+1) = "       - S12_mag: "   + string(c.S12_mag) + ", S12_phase: " + string(c.S12_phase);
                infoLines(end+1) = "       - S22_mag: "   + string(c.S22_mag) + ", S22_phase: " + string(c.S22_phase);
            end
        end
    end

    infoLines(end+1) = "";
    infoLines(end+1) = "Output files:";
    infoLines(end+1) = "   - Permittivity merged:         *" + permMergedSuffix;
    infoLines(end+1) = "   - Permittivity averaged:       *" + permAveragedSuffix;
    infoLines(end+1) = "   - S2P merged:                  *" + s2pMergedSuffix;
    infoLines(end+1) = "   - S2P averaged:                *" + s2pAveragedSuffix;
    infoLines(end+1) = "";
    infoLines(end+1) = "Click 'Continue' to proceed or 'Cancel' to exit.";
end
