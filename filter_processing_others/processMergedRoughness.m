function processMergedRoughness()
    % ============================================================
    % Roughness & WithoutRoughness & S2P Averaging Script
    %
    % - Root klasör: uigetdir ile seçiliyor.
    % - Altındaki TÜM "Merged_Revision" klasörleri bulunuyor.
    % - Her Merged_Revision içinde şu dosya tipleri aranıyor:
    %
    %   1) *_Merged_Data_Roughness.csv
    %      -> *_Averaged_Data_Roughness.csv
    %
    %   2) *_Merged_Data_withoutRoughness.csv
    %      -> *_Averaged_Data_withoutRoughness.csv
    %
    %   3) *_S2PMerged_RoughnessS2P.csv
    %      -> *_S2PAveraged_RoughnessS2P.csv
    %
    %   4) *_S2PMerged_withoutRoughnessS2P.csv
    %      -> *_S2PAveraged_withoutRoughnessS2P.csv
    %
    % - Permittivity merged format (4 kolon bloklar):
    %      [Freq, EpsReal, TanDelta, Thickness] × Nsample
    %
    % - S2P merged format (3 kolon bloklar):
    %      [Freq, EpsReal, TanDelta] × Nsample
    %
    % - Çıkışlar:
    %      6 kolon:
    %        freq, epsilon_r, std_epsilon_r,
    %        tan_delta, std_tan_delta, thickness
    %      (S2P tarafında thickness = NaN)
    % ============================================================

    % ------------------------------------------------------------
    % 0) JSON'dan hangi işleri yapacağımızı ve suffix'leri oku
    % ------------------------------------------------------------
    configPath = 'config_VNA.json';
    if ~isfile(configPath)
        error('Config file not found: %s', configPath);
    end
    cfgText = fileread(configPath);
    cfg     = jsondecode(cfgText);

    % FilePatterns ve Debug bloklarını kullanacağız
    if ~isfield(cfg, "FilePatterns")
        error('Config missing "FilePatterns" block.');
    end
    FP  = cfg.FilePatterns;

    if ~isfield(cfg, "Debug")
        DBG.EnableRoughness      = true;
        DBG.SaveWithRoughness    = true;
        DBG.SaveWithoutRoughness = true;
    else
        DBG = cfg.Debug;
    end

    % MergedDirectory ismi (ör: "Merged_Revision")
    if isfield(FP, "MergedDirectory") && ~isempty(FP.MergedDirectory)
        mergedDirName = FP.MergedDirectory;
        if isstring(mergedDirName), mergedDirName = char(mergedDirName); end
    else
        mergedDirName = 'Merged_Revision';
    end

    % Perm için suffix'ler
    roughSuffixPerm   = getFieldOrDefault(FP, 'OutputFileSuffix',    '_Roughness');
    noRoughSuffixPerm = getFieldOrDefault(FP, 'OutputFileSuffix1',   '_withoutRoughness');

    % S2P için suffix'ler
    roughSuffixS2P    = getFieldOrDefault(FP, 'OutputFileSuffixS2P',  '_RoughnessS2P');
    noRoughSuffixS2P  = getFieldOrDefault(FP, 'OutputFileSuffix1S2P', '_withoutRoughnessS2P');

    % Perm merged temel token (örn: "*Merged_Data.csv" → "Merged_Data")
    if isfield(FP, "MergedFilePattern") && ~isempty(FP.MergedFilePattern)
        pat = FP.MergedFilePattern;            % örn: "*Merged_Data.csv"
        if isstring(pat), pat = char(pat); end
        pat = strrep(pat, '*', '');
        [~, basePermToken, ~] = fileparts(pat);% "Merged_Data"
    else
        basePermToken = 'Merged_Data';
    end

    % S2P merged temel token (örn: "_S2PMerged.csv" → "S2PMerged")
    s2pBaseToken = 'S2PMerged';
    if isfield(cfg, "MergedProcessing") && ...
       isfield(cfg.MergedProcessing, "Output") && ...
       isfield(cfg.MergedProcessing.Output, "S2PMergedSuffix")

        s2pSuffixCfg = cfg.MergedProcessing.Output.S2PMergedSuffix;   % "_S2PMerged.csv"
        if isstring(s2pSuffixCfg), s2pSuffixCfg = char(s2pSuffixCfg); end
        s2pSuffixCfg = strrep(s2pSuffixCfg, '.csv', '');  % "_S2PMerged"
        s2pSuffixCfg = strrep(s2pSuffixCfg, '_', '');     % "S2PMerged"
        s2pBaseToken = s2pSuffixCfg;
    end

    % Hangi işleri yapalım?
    doPermRough   = isTrue(DBG, 'EnableRoughness')   && isTrue(DBG, 'SaveWithRoughness');
    doPermNoRough = isTrue(DBG, 'SaveWithoutRoughness');
    doS2PRough    = doPermRough;
    doS2PNoRough  = doPermNoRough;

    if ~(doPermRough || doPermNoRough || doS2PRough || doS2PNoRough)
        disp('Nothing to do. Check Debug.EnableRoughness / SaveWithRoughness / SaveWithoutRoughness in JSON.');
        return;
    end

    % ------------------------------------------------------------
    % 1) Root klasör seçimi (eski çalışan yapı gibi)
    % ------------------------------------------------------------
    rootDir = uigetdir([], 'Select the ROOT folder that contains the "Merged_Revision" folders');
    if isequal(rootDir, 0)
        disp('Operation cancelled by user.');
        return;
    end

    msg = sprintf([ ...
        'This script will scan every "%s" folder under:\n\n%s\n\n', ...
        'and process:\n', ...
        '  - *_Merged_Data_Roughness.csv          → *_Averaged_Data_Roughness.csv\n', ...
        '  - *_Merged_Data_withoutRoughness.csv   → *_Averaged_Data_withoutRoughness.csv\n', ...
        '  - *_S2PMerged_RoughnessS2P.csv         → *_S2PAveraged_RoughnessS2P.csv\n', ...
        '  - *_S2PMerged_withoutRoughnessS2P.csv  → *_S2PAveraged_withoutRoughnessS2P.csv\n\n', ...
        'Click "Continue" to start.' ], ...
        mergedDirName, rootDir);

    userChoice = questdlg(msg, 'Roughness Averaging', 'Continue', 'Cancel', 'Continue');
    if ~strcmp(userChoice, 'Continue')
        disp('Operation cancelled by user.');
        return;
    end

    % ------------------------------------------------------------
    % 2) Tüm ".../Merged_Revision" dizinlerini bul (eski mantık)
    % ------------------------------------------------------------
    mergedDirs = dir(fullfile(rootDir, '**', mergedDirName));
    mergedDirs = mergedDirs([mergedDirs.isdir]);
    mergedDirs = mergedDirs(~ismember({mergedDirs.name}, {'..'}));

    if isempty(mergedDirs)
        fprintf('No "%s" folders found under %s\n', mergedDirName, rootDir);
        return;
    end

    % ------------------------------------------------------------
    % 3) Her Merged_Revision klasörünü işle
    % ------------------------------------------------------------
    for i = 1:numel(mergedDirs)
        curDir = fullfile(mergedDirs(i).folder, mergedDirs(i).name);
        fprintf('\n=== Folder: %s ===\n', curDir);

        % --- 3.1 | Permittivity – Roughness ----------------------
        if doPermRough
            patternPermR = ['*' basePermToken roughSuffixPerm '.csv'];   % örn: *Merged_Data_Roughness.csv
            processPermSet(curDir, patternPermR, ...
                ['_' basePermToken roughSuffixPerm '.csv'], ...
                ['_Averaged_Data' roughSuffixPerm '.csv']);
        end

        % --- 3.2 | Permittivity – Without Roughness --------------
        if doPermNoRough
            patternPermNR = ['*' basePermToken noRoughSuffixPerm '.csv']; % *Merged_Data_withoutRoughness.csv
            processPermSet(curDir, patternPermNR, ...
                ['_' basePermToken noRoughSuffixPerm '.csv'], ...
                ['_Averaged_Data' noRoughSuffixPerm '.csv']);
        end

        % --- 3.3 | S2P – Roughness (3 kolon blok, 6 kolon çıkış) --
        if doS2PRough
            patternS2PR = ['*' s2pBaseToken roughSuffixS2P '.csv'];       % *S2PMerged_RoughnessS2P.csv
            processS2PSet(curDir, patternS2PR, ...
                ['_' s2pBaseToken roughSuffixS2P '.csv'], ...
                ['_S2PAveraged' roughSuffixS2P '.csv']);
        end

        % --- 3.4 | S2P – Without Roughness ------------------------
        if doS2PNoRough
            patternS2PNR = ['*' s2pBaseToken noRoughSuffixS2P '.csv'];    % *S2PMerged_withoutRoughnessS2P.csv
            processS2PSet(curDir, patternS2PNR, ...
                ['_' s2pBaseToken noRoughSuffixS2P '.csv'], ...
                ['_S2PAveraged' noRoughSuffixS2P '.csv']);
        end
    end

    fprintf('\n...Completed...\n');
end

% =====================================================================
% Yardımcı: Perm setini işle (4 kolon blok: F, eps, tan, thick)
% =====================================================================
function processPermSet(curDir, pattern, inTag, outTag)
    inFiles = dir(fullfile(curDir, pattern));

    if isempty(inFiles)
        fprintf('  [Perm %s] No files found matching %s\n', outTag, pattern);
        return;
    end

    fprintf('  [Perm %s] %d file(s) found.\n', outTag, numel(inFiles));

    for f = 1:numel(inFiles)
        inPath = fullfile(inFiles(f).folder, inFiles(f).name);
        fprintf('    → %s\n', inFiles(f).name);

        try
            T    = readtable(inPath, 'VariableNamingRule','preserve');
            data = T{:,:};
            [nRows, nCols] = size(data);

            % 4 kolon blok kontrolü
            if mod(nCols, 4) ~= 0
                warning('      ! Column count is not a multiple of 4. Skipped.');
                continue;
            end

            nSamples = nCols / 4;

            % Her 4. kolon frekans: 1,5,9,...
            freqMat   = data(:, 1:4:nCols);
            freqCheck = all(abs(freqMat - freqMat(:,1)) < 1e-12 | isnan(freqMat), 1);

            if all(freqCheck)
                useCols = 1:nCols;
            elseif all(freqCheck(1:min(3, size(freqCheck,2))))
                warning('      ! Not all frequency columns match. Using only first 3 samples.');
                % İlk 3 sample = 12 kolon
                useCols = 1:12;
            else
                warning('      ! Frequency mismatch even among first 3 samples. Skipped.');
                continue;
            end

            % Frekans & matrisler
            freq     = data(:, useCols(1));
            epsMat   = data(:, useCols(2:4:end));
            tanMat   = data(:, useCols(3:4:end));
            thickMat = data(:, useCols(4:4:end));

            % Negatif eps'leri ignore et
            badRows = epsMat < 0;
            epsMat(badRows) = NaN;
            tanMat(badRows) = NaN;

            epsAvg = mean(epsMat,   2, 'omitnan');
            epsStd = std(epsMat,    0, 2, 'omitnan');
            tanAvg = mean(tanMat,   2, 'omitnan');
            tanStd = std(tanMat,    0, 2, 'omitnan');
            thkAvg = mean(thickMat, 2, 'omitnan');

            outTbl = table(freq, epsAvg, epsStd, tanAvg, tanStd, thkAvg, ...
                'VariableNames', {'freq','epsilon_r','std_epsilon_r', ...
                                  'tan_delta','std_tan_delta','thickness'});

            % Çıkış yolu: suffix değiştir
            outPath = strrep(inPath, inTag, outTag);

            writetable(outTbl, outPath);
            fprintf('      → Saved %s\n', outPath);

        catch ME
            fprintf('      ! Error: %s  (%s)\n', inPath, ME.message);
        end
    end
end

% =====================================================================
% Yardımcı: S2P setini işle (3 kolon blok: F, eps, tan; 6 kolon çıkış)
% =====================================================================
function processS2PSet(curDir, pattern, inTag, outTag)
    inFiles = dir(fullfile(curDir, pattern));

    if isempty(inFiles)
        fprintf('  [S2P %s] No files found matching %s\n', outTag, pattern);
        return;
    end

    fprintf('  [S2P %s] %d file(s) found.\n', outTag, numel(inFiles));

    for f = 1:numel(inFiles)
        inPath = fullfile(inFiles(f).folder, inFiles(f).name);
        fprintf('    → %s\n', inFiles(f).name);

        try
            T    = readtable(inPath, 'VariableNamingRule','preserve');
            data = T{:,:};
            [nRows, nCols] = size(data);

            % 3 kolon blok kontrolü
            if mod(nCols, 3) ~= 0
                warning('      ! Column count is not a multiple of 3 (F,eps,tan). Skipped.');
                continue;
            end

            nSamples = nCols / 3;

            % Her 3. kolon frekans: 1,4,7,...
            freqMat   = data(:, 1:3:nCols);
            freqCheck = all(abs(freqMat - freqMat(:,1)) < 1e-12 | isnan(freqMat), 1);

            if all(freqCheck)
                useSamples = 1:nSamples;
            elseif nSamples >= 3 && all(freqCheck(1:3))
                warning('      ! Not all frequency columns match. Using only first 3 samples.');
                useSamples = 1:3;
            else
                warning('      ! Frequency mismatch even among first 3 samples. Skipped.');
                continue;
            end

            % Frekans ve eps/tan matrisleri
            nUsed = numel(useSamples);
            freq   = data(:, (useSamples(1)-1)*3 + 1);
            epsMat = nan(nRows, nUsed);
            tanMat = nan(nRows, nUsed);

            for idx = 1:nUsed
                s     = useSamples(idx);
                baseC = (s-1)*3;
                epsMat(:, idx) = data(:, baseC + 2);
                tanMat(:, idx) = data(:, baseC + 3);
            end

            % Negatif eps'leri ignore et
            badRows = epsMat < 0;
            epsMat(badRows) = NaN;
            tanMat(badRows) = NaN;

            epsAvg = mean(epsMat, 2, 'omitnan');
            epsStd = std(epsMat,  0, 2, 'omitnan');
            tanAvg = mean(tanMat, 2, 'omitnan');
            tanStd = std(tanMat,  0, 2, 'omitnan');

            % Thickness: S2P için NaN
            thkAvg = NaN(nRows, 1);

            outTbl = table(freq, epsAvg, epsStd, tanAvg, tanStd, thkAvg, ...
                'VariableNames', {'freq','epsilon_r','std_epsilon_r', ...
                                  'tan_delta','std_tan_delta','thickness'});

            % Çıkış yolu: suffix değiştir
            outPath = strrep(inPath, inTag, outTag);

            writetable(outTbl, outPath);
            fprintf('      → Saved %s\n', outPath);

        catch ME
            fprintf('      ! Error: %s  (%s)\n', inPath, ME.message);
        end
    end
end

% =====================================================================
% Küçük yardımcılar
% =====================================================================
function val = getFieldOrDefault(S, name, defaultVal)
    if isfield(S, name) && ~isempty(S.(name))
        v = S.(name);
        if isstring(v), v = char(v); end
        val = v;
    else
        val = defaultVal;
    end
end

function tf = isTrue(S, fieldName)
    tf = false;
    if ~isfield(S, fieldName), return; end
    v = S.(fieldName);
    if islogical(v)
        tf = v;
    elseif isnumeric(v)
        tf = (v ~= 0);
    elseif ischar(v) || isstring(v)
        v = lower(char(v));
        tf = any(strcmp(v, {'1','true','yes'}));
    end
end
