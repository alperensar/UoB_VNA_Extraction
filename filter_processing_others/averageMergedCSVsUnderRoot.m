function averageMergedCSVsUnderRoot(action)
% averageMergedCSVsUnderRoot(action)
%
% ACTIONS:
%   'preview_cleanup' : lists wrongly generated *AveragedBands*.csv files
%   'delete_cleanup'  : lists + asks confirmation + deletes those files
%   'average'         : averages all *Merged*.csv files inside chosen source folders
%
% Fixes:
%   - Never renames "MergedBands" -> "AveragedBands"
%   - Renames only tokens:
%       _S2PMerged   -> _S2PAveraged
%       _Merged_Data -> _Averaged_Data
%       _Merged.csv  -> _Averaged.csv   (fallback, end-only)
%   - Does NOT output Thickness mean/std (skips Thickness columns entirely)

    if nargin < 1 || isempty(action)
        action = questdlg( ...
            ['Select operation:' newline newline ...
             '  • Preview cleanup list (AveragedBands files)' newline ...
             '  • Delete those files' newline ...
             '  • Run averaging'], ...
            'Operation', ...
            'Preview cleanup list', ...
            'Delete cleanup files', ...
            'Run averaging', ...
            'Preview cleanup list');

        if isempty(action)
            disp('Operation cancelled.');
            return;
        end

        if strcmpi(action, 'Preview cleanup list')
            action = 'preview_cleanup';
        elseif strcmpi(action, 'Delete cleanup files')
            action = 'delete_cleanup';
        else
            action = 'average';
        end
    end

    switch lower(string(action))
        case "preview_cleanup"
            rootDir = uigetdir([], 'Select your MAIN root folder (e.g. RAWDATA)');
            if isequal(rootDir, 0), disp('Cancelled.'); return; end
            files = findWrongAveragedBandsFiles(rootDir);
            printFileList(files, 'FILES THAT WOULD BE DELETED (*AveragedBands*.csv)');
            fprintf('\nTip: if this list looks correct, run:\n  averageMergedCSVsUnderRoot(''delete_cleanup'')\n');
            return;

        case "delete_cleanup"
            rootDir = uigetdir([], 'Select your MAIN root folder (e.g. RAWDATA)');
            if isequal(rootDir, 0), disp('Cancelled.'); return; end
            files = findWrongAveragedBandsFiles(rootDir);
            printFileList(files, 'FILES MARKED FOR DELETION (*AveragedBands*.csv)');

            if isempty(files)
                disp('No files to delete.');
                return;
            end

            answ = questdlg( ...
                sprintf('Delete %d files listed in the Command Window?', numel(files)), ...
                'Confirm deletion', ...
                'Delete', 'Cancel', 'Cancel');

            if ~strcmpi(answ, 'Delete')
                disp('Deletion cancelled.');
                return;
            end

            nDel = 0;
            for i = 1:numel(files)
                try
                    delete(fullfile(files(i).folder, files(i).name));
                    nDel = nDel + 1;
                catch ME
                    fprintf('  !! Could not delete: %s (%s)\n', fullfile(files(i).folder, files(i).name), ME.message);
                end
            end
            fprintf('Deleted %d/%d files.\n', nDel, numel(files));
            return;

        case "average"
            % continue below
        otherwise
            error('Unknown action: %s', string(action));
    end

    % -------------------------
    % AVERAGING MODE
    % -------------------------
    rootDir = uigetdir([], 'Select your MAIN root folder (e.g. RAWDATA)');
    if isequal(rootDir, 0)
        disp('Operation cancelled (no root folder selected).');
        return;
    end

    srcChoice = questdlg( ...
        ['Which source folder should be used?' newline newline ...
         '  • Merged' newline ...
         '  • Merged_Filtered_Revision'], ...
        'Source folder selection', ...
        'Merged', 'Merged_Filtered_Revision', 'Merged');

    if isempty(srcChoice)
        disp('Operation cancelled (no source folder selected).');
        return;
    end
    srcFolderName = char(srcChoice);

    fprintf('\n=== ROOT: %s\n', rootDir);
    fprintf('=== SOURCE FOLDER: %s\n\n', srcFolderName);

    srcDirs = dir(fullfile(rootDir, '**', srcFolderName));
    srcDirs = srcDirs([srcDirs.isdir]);

    if isempty(srcDirs)
        fprintf('No "%s" folders found under root.\n', srcFolderName);
        return;
    end

    totalFiles = 0;
    okFiles    = 0;
    skipFiles  = 0;
    errFiles   = 0;

    for d = 1:numel(srcDirs)
        thisFolder = fullfile(srcDirs(d).folder, srcDirs(d).name);

        csvFiles = dir(fullfile(thisFolder, '*.csv'));
        if isempty(csvFiles)
            continue;
        end

        for k = 1:numel(csvFiles)
            inName = csvFiles(k).name;

            % Only files whose filename contains "Merged"
            if ~contains(inName, 'Merged', 'IgnoreCase', true)
                continue;
            end

            % Skip obviously wrong old outputs + already averaged
            if contains(inName, 'AveragedBands', 'IgnoreCase', true) || ...
               contains(inName, 'Averaged',      'IgnoreCase', true)
                skipFiles = skipFiles + 1;
                continue;
            end

            totalFiles = totalFiles + 1;

            inPath = fullfile(csvFiles(k).folder, inName);

            outName = makeAveragedFilenameSafe(inName);
            outPath = fullfile(csvFiles(k).folder, outName);

            fprintf('Processing: %s\n', inPath);

            try
                T = readtable(inPath, 'VariableNamingRule','preserve');
                data = T{:,:};

                if ~isnumeric(data)
                    warning('  -> Non-numeric table detected. Skipping: %s', inName);
                    skipFiles = skipFiles + 1;
                    continue;
                end

                Tout = buildAveragedTableFromMergedTable_NoThickness(T);

                writetable(Tout, outPath);
                fprintf('  -> Saved: %s\n', outPath);
                okFiles = okFiles + 1;

            catch ME
                fprintf('  !! ERROR: %s\n', ME.message);
                errFiles = errFiles + 1;
            end
        end
    end

    fprintf('\nDONE.\n');
    fprintf('  Total Merged CSVs found: %d\n', totalFiles);
    fprintf('  Averaged written:        %d\n', okFiles);
    fprintf('  Skipped:                 %d\n', skipFiles);
    fprintf('  Errors:                  %d\n', errFiles);
end


% ============================================================
% Find wrong files: *AveragedBands*.csv inside Merged_Filtered_Revision
% ============================================================
function files = findWrongAveragedBandsFiles(rootDir)
    files = dir(fullfile(rootDir, '**', 'Merged_Filtered_Revision', '*AveragedBands*.csv'));
end

function printFileList(files, titleStr)
    fprintf('\n=== %s ===\n', titleStr);
    if isempty(files)
        fprintf('  (none)\n');
        return;
    end
    for i = 1:numel(files)
        fprintf('  [%4d] %s\n', i, fullfile(files(i).folder, files(i).name));
    end
end


% ============================================================
% Safe rename: DO NOT touch "MergedBands"
% Only change the true merged-token part
% ============================================================
function outName = makeAveragedFilenameSafe(inName)
    outName = inName;

    % 1) S2P: _S2PMerged -> _S2PAveraged
    outName = regexprep(outName, '(?i)_S2PMerged', '_S2PAveraged');

    % 2) Perm: _Merged_Data -> _Averaged_Data
    outName = regexprep(outName, '(?i)_Merged_Data', '_Averaged_Data');

    % 3) Fallback (end-only): ..._Merged.csv -> ..._Averaged.csv
    outName = regexprep(outName, '(?i)_Merged(\.csv)$', '_Averaged$1');

    % Important: this NEVER changes "MergedBands" because we never replace plain "Merged"
end


% ============================================================
% Averaging builder: skip Thickness entirely (no mean/std)
% ============================================================
function Tout = buildAveragedTableFromMergedTable_NoThickness(T)
    varNames = T.Properties.VariableNames;
    data     = T{:,:};
    nCols    = numel(varNames);

    runIdx  = nan(1, nCols);
    baseStr = cell(1, nCols);

    for c = 1:nCols
        [ri, bn] = parseReplicatedHeader(varNames{c});
        runIdx(c)  = ri;
        baseStr{c} = bn;
    end

    hasRunPrefix = any(isfinite(runIdx));

    if hasRunPrefix
        baseLower = lower(string(baseStr));
        [~, ia]   = unique(baseLower, 'stable');
        bases     = string(baseStr(ia));
    else
        bases = detectBasesByRepeatingPattern(varNames);
        if isempty(bases)
            error('Could not detect replicate blocks from headers.');
        end
    end

    basesLower = lower(bases);
    iFreq = find(contains(basesLower, "freq"), 1);
    if isempty(iFreq)
        error('No frequency-like column found (no "freq" in base names).');
    end
    freqBase = char(bases(iFreq));

    freqCols = find(strcmpi(string(baseStr), string(freqBase)));
    if isempty(freqCols)
        error('Frequency base detected, but no columns match it.');
    end
    freqVec = data(:, freqCols(1));

    outVars = {};
    outData = [];

    outVars{end+1} = makeValidNameSmart(freqBase);
    outData(:, end+1) = freqVec;

    for b = 1:numel(bases)
        baseName = char(bases(b));

        if strcmpi(baseName, freqBase)
            continue;
        end

        % Skip Thickness columns completely
        if contains(lower(baseName), 'thickness')
            continue;
        end

        cols = find(strcmpi(string(baseStr), string(baseName)));
        if isempty(cols)
            continue;
        end

        X  = data(:, cols);
        mu = mean(X, 2, 'omitnan');
        sd = std(X,  0, 2, 'omitnan');

        outVars{end+1} = makeValidNameSmart([baseName '_mean']); %#ok<AGROW>
        outData(:, end+1) = mu; %#ok<AGROW>

        outVars{end+1} = makeValidNameSmart([baseName '_std']); %#ok<AGROW>
        outData(:, end+1) = sd; %#ok<AGROW>
    end

    Tout = array2table(outData, 'VariableNames', outVars);
end


function [runIndex, baseName] = parseReplicatedHeader(headerName)
    runIndex = NaN;
    baseName = headerName;

    tok = regexp(headerName, '^x?(\d+)[\_\-](.+)$', 'tokens', 'once');
    if ~isempty(tok)
        runIndex = str2double(tok{1});
        baseName = tok{2};
    end
end

function bases = detectBasesByRepeatingPattern(varNames)
    bases = string.empty;
    if isempty(varNames), return; end

    first = string(varNames{1});
    reps  = find(strcmpi(string(varNames), first));

    if numel(reps) < 2
        return;
    end

    blockSize = reps(2) - reps(1);
    if blockSize <= 0 || blockSize > numel(varNames)
        return;
    end

    bases = string(varNames(1:blockSize));
end

function s = makeValidNameSmart(raw)
    s = matlab.lang.makeValidName(char(raw), 'ReplacementStyle','delete');
    if isempty(s)
        s = 'Var';
    end
end
