% function averagedDataLast()
% % averagedDataLast
% % Uses filtered merged data (Merged_Filtered_Revision) to build final
% % averaged datasets for permittivity and S2P, and exports them into:
% %
% %   - AveragedLast_Revision        (permittivity)
% %   - AveragedLast_Revision_S2P    (S2P)
% %
% % Behaviour:
% %   - For Keysight_PNA_UoB: only permittivity exists. The same perm
% %     outputs are written to BOTH AveragedLast_Revision and
% %     AveragedLast_Revision_S2P.
% %
% %   - For Keysight_PNA_NPL:
% %       * Permittivity averaged files
% %         (_Averaged_Data_Roughness.csv / _Averaged_Data_withoutRoughness.csv)
% %         → AveragedLast_Revision
% %       * S2P averaged files
% %         (_S2PAveraged_RoughnessS2P.csv / _S2PAveraged_withoutRoughnessS2P.csv)
% %         → AveragedLast_Revision_S2P
% %
% % Old behaviour preserved:
% %   - Band ordering and stitching
% %   - Gap-based segmentation (>2 GHz)
% %   - CSV + MAT export
% %   - Shaded-error plots for eps' and tanδ
% 
%     %------------------------------------------------------
%     % 1) Root folder seçimi
%     %------------------------------------------------------
%     rootDir = uigetdir([], 'Select your root folder (RAWDATA)');
%     if rootDir == 0
%         disp('Operation cancelled by user.');
%         return;
%     end
% 
%     % İki ölçüm metodu
%     measurementMethods = {'Keysight_PNA_NPL','Keysight_PNA_UoB'};
% 
%     % Aranan dosya sonları (suffix)
%     permSuffixRough = '_Averaged_Data_Roughness.csv';
%     permSuffixNo    = '_Averaged_Data_withoutRoughness.csv';
% 
%     s2pSuffixRough  = '_S2PAveraged_RoughnessS2P.csv';
%     s2pSuffixNo     = '_S2PAveraged_withoutRoughnessS2P.csv';
% 
%     % Sample klasörleri
%     sampleFolders = dir(rootDir);
%     sampleFolders = sampleFolders([sampleFolders.isdir] & ...
%                                   ~startsWith({sampleFolders.name}, '.'));
% 
%     for s = 1:numel(sampleFolders)
%         sampleName       = sampleFolders(s).name;
%         sampleFolderPath = fullfile(rootDir, sampleName);
% 
%         for m = 1:numel(measurementMethods)
%             methodName       = measurementMethods{m};
%             methodFolderPath = fullfile(sampleFolderPath, methodName);
% 
%             if ~exist(methodFolderPath,'dir')
%                 continue;
%             end
% 
%             %--------------------------------------------------
%             % 2) Band isimleri ve cutoff frekansları
%             %--------------------------------------------------
%             if strcmp(methodName,'Keysight_PNA_UoB')
%                 freqBandsOrdered = {'220-330','325-500','500-750', 'MergedBands'};
%                 cutoffFreqsHz    = [325e9, 500e9];
%             else
%                 freqBandsOrdered = {'WR15','WR10','WR5','WM-380', 'MergedBands'};
%                 cutoffFreqsHz    = [75e9, 110e9, 220e9];
%             end
% 
%             % Çıktı klasörleri
%             outPerm = fullfile(methodFolderPath,'AveragedLast_Revision');
%             outS2P  = fullfile(methodFolderPath,'AveragedLast_Revision_S2P');
%             if ~exist(outPerm,'dir'), mkdir(outPerm); end
%             if ~exist(outS2P,'dir'),  mkdir(outS2P);  end
% 
%             %--------------------------------------------------
%             % 3) Method klasörü altındaki band klasörlerini tara
%             %--------------------------------------------------
%             bandFolders = dir(methodFolderPath);
%             bandFolders = bandFolders([bandFolders.isdir] & ...
%                                       ~startsWith({bandFolders.name}, '.'));
% 
%             % Toplayacağımız dosya listeleri
%             permRoughFiles = [];
%             permNoFiles    = [];
%             s2pRoughFiles  = [];
%             s2pNoFiles     = [];
% 
%             for b = 1:numel(bandFolders)
%                 bandName = bandFolders(b).name;
% 
%                 % Sadece tanımlanmış band adlarını al
%                 if ~ismember(bandName, freqBandsOrdered)
%                     continue;
%                 end
% 
%                 bandFolderPath = fullfile(methodFolderPath, bandName);
% 
%                 % Bu band altında Merged_Filtered_Revision var mı?
%                 srcDir = fullfile(bandFolderPath, 'Merged_Filtered_Revision');
%                 if ~exist(srcDir, 'dir')
%                     continue;
%                 end
% 
%                 % Bu klasördeki tüm CSV dosyaları
%                 csvFiles = dir(fullfile(srcDir, '*.csv'));
%                 if isempty(csvFiles)
%                     continue;
%                 end
% 
%                 % Dosya adlarına göre ayır (suffix ile)
%                 for k = 1:numel(csvFiles)
%                     fname = csvFiles(k).name;
%                     csvFiles(k).folder = srcDir;  % tam path için
% 
%                     if endsWith(fname, permSuffixRough, 'IgnoreCase', true)
%                         permRoughFiles = [permRoughFiles; csvFiles(k)];
%                     elseif endsWith(fname, permSuffixNo, 'IgnoreCase', true)
%                         permNoFiles    = [permNoFiles; csvFiles(k)];
%                     elseif endsWith(fname, s2pSuffixRough, 'IgnoreCase', true)
%                         s2pRoughFiles  = [s2pRoughFiles; csvFiles(k)];
%                     elseif endsWith(fname, s2pSuffixNo, 'IgnoreCase', true)
%                         s2pNoFiles     = [s2pNoFiles; csvFiles(k)];
%                     end
%                 end
%             end
% 
%             %--------------------------------------------------
%             % 4) PERMITTIVITY (NPL + UoB)
%             %    Roughness ve WithoutRoughness ayrı ayrı
%             %--------------------------------------------------
%             if ~isempty(permRoughFiles)
%                 fprintf('\n📘 PERM Roughness: %s | %s\n', sampleName, methodName);
%                 processAndSaveAveraged(permRoughFiles, freqBandsOrdered, ...
%                     outPerm, sampleName, 'Averaged_Roughness', cutoffFreqsHz, '_roughness');
% 
%                 % UoB ise: aynı dataset S2P klasörüne de kopyalanacak
%                 if strcmp(methodName,'Keysight_PNA_UoB')
%                     processAndSaveAveraged(permRoughFiles, freqBandsOrdered, ...
%                         outS2P, sampleName, 'Averaged_Roughness', cutoffFreqsHz, '_roughness');
%                 end
%             end
% 
%             if ~isempty(permNoFiles)
%                 fprintf('\n📘 PERM WithoutRough: %s | %s\n', sampleName, methodName);
%                 processAndSaveAveraged(permNoFiles, freqBandsOrdered, ...
%                     outPerm, sampleName, 'Averaged', cutoffFreqsHz, '');
% 
%                 if strcmp(methodName,'Keysight_PNA_UoB')
%                     processAndSaveAveraged(permNoFiles, freqBandsOrdered, ...
%                         outS2P, sampleName, 'Averaged', cutoffFreqsHz, '');
%                 end
%             end
% 
%             %--------------------------------------------------
%             % 5) S2P (yalnızca NPL tarafında anlamlı)
%             %--------------------------------------------------
%             if ~strcmp(methodName,'Keysight_PNA_NPL')
%                 % UoB için S2P veri yok; atla
%                 continue;
%             end
% 
%             if ~isempty(s2pRoughFiles)
%                 fprintf('\n📗 S2P Roughness: %s | %s\n', sampleName, methodName);
%                 processAndSaveAveraged(s2pRoughFiles, freqBandsOrdered, ...
%                     outS2P, sampleName, 'S2PAveraged_RoughnessS2P', cutoffFreqsHz, '_roughnessS2P');
%             end
% 
%             if ~isempty(s2pNoFiles)
%                 fprintf('\n📗 S2P WithoutRough: %s | %s\n', sampleName, methodName);
%                 processAndSaveAveraged(s2pNoFiles, freqBandsOrdered, ...
%                     outS2P, sampleName, 'S2PAveraged_withoutRoughnessS2P', cutoffFreqsHz, '_withoutRoughnessS2P');
%             end
% 
%         end
%     end
% 
%     fprintf('\n✅ All samples / methods processed.\n');
% end
% 
% % =====================================================================
% % Averaged dosyaları işleme:
% %  - fileList          : Averaged_Data / S2PAveraged_* listesi
% %  - freqBandsOrdered  : band isimleri sırası
% %  - outputFolder      : çıktı klasörü
% %  - sampleName        : örnek ismi
% %  - dataType          : çıktı ismi için (Averaged, Averaged_Roughness, ...)
% %  - cutoffFreqsHz     : band geçişleri için referans (plot'ta kullanılabilir)
% %  - suffix            : plot dosya ismine eklenecek ek (örn: _roughness)
% % =====================================================================
% function processAndSaveAveraged(fileList, freqBandsOrdered, outputFolder, sampleName, dataType, cutoffFreqsHz, suffix)
%     if isempty(fileList)
%         return;
%     end
% 
%     fprintf('   → Processing %s (%d files)\n', dataType, numel(fileList));
% 
%     dataCell   = cell(numel(fileList), 1);
%     headerCell = cell(numel(fileList), 1);
%     bandIndex  = zeros(numel(fileList), 1);
% 
%     %--------------------------------------------------
%     % 1) Dosyaları oku ve band indeksini belirle
%     %--------------------------------------------------
%     for j = 1:numel(fileList)
%         fPath    = fullfile(fileList(j).folder, fileList(j).name);
%         cellData = readcell(fPath);
% 
%         headerCell{j} = cellData(1,:);
%         dataOnly      = cellData(2:end,:);
% 
%         % Numeric mat
%         numericMatrix = cell2mat(cellfun(@double, dataOnly, 'UniformOutput', false));
%         dataCell{j}   = numericMatrix;
% 
%         % Dosya adından bandName çıkart (örn: ..._220-330_Averaged_Data_Roughness.csv)
%         nameParts = split(fileList(j).name, '_');
%         if numel(nameParts) >= 3
%             bandName = nameParts{end-2};
%         else
%             bandName = '';
%         end
% 
%         idx = find(strcmp(freqBandsOrdered, bandName), 1);
%         if isempty(idx)
%             idx = Inf;  % tanımsız band en sona gider
%         end
%         bandIndex(j) = idx;
%     end
% 
%     %--------------------------------------------------
%     % 2) Band sırasına göre sort et
%     %--------------------------------------------------
%     [~, sortIdx]   = sort(bandIndex);
%     sortedDataCell = dataCell(sortIdx);
%     headerCell     = headerCell(sortIdx);
% 
%     %--------------------------------------------------
%     % 3) Tek büyük combinedData oluştur
%     %--------------------------------------------------
%     combinedData = vertcat(sortedDataCell{:});
%     totalRows    = size(combinedData,1);
% 
%     %--------------------------------------------------
%     % 4) Segmentler:
%     %    - Her band arası (sortedDataCell ile)
%     %    - Frekans farkı > 2GHz noktaları
%     %--------------------------------------------------
%     numBands      = numel(sortedDataCell);
%     segmentStarts = zeros(numBands,1);
%     segmentEnds   = zeros(numBands,1);
% 
%     rowCounter = 1;
%     for i = 1:numBands
%         nRows           = size(sortedDataCell{i},1);
%         segmentStarts(i)= rowCounter;
%         segmentEnds(i)  = rowCounter + nRows - 1;
%         rowCounter      = rowCounter + nRows;
%     end
% 
%     % Frekans farklarına göre gap noktaları
%     freqDiff        = diff(combinedData(:,1));
%     gapPoints       = find(freqDiff > 2e9);
%     gapSegmentStarts= gapPoints + 1;
%     gapSegmentEnds  = gapPoints;
% 
%     % Tüm segmentleri birleştir
%     allSegmentStarts = [segmentStarts; gapSegmentStarts(:)];
%     allSegmentEnds   = [segmentEnds;   gapSegmentEnds(:)];
% 
%     [allSegmentStarts, ia] = sort(allSegmentStarts);
%     allSegmentEnds         = allSegmentEnds(ia);
% 
%     validIdx = allSegmentStarts <= totalRows & ...
%                allSegmentEnds   <= totalRows & ...
%                allSegmentStarts <= allSegmentEnds;
% 
%     segmentStarts = allSegmentStarts(validIdx);
%     segmentEnds   = allSegmentEnds(validIdx);
% 
%     %--------------------------------------------------
%     % 5) CSV + MAT kaydet
%     %--------------------------------------------------
%     fileNameOut = regexprep(sampleName, '\s+', '_');  % boşlukları altçizgi yap
%     baseName    = sprintf('%s_%s', fileNameOut, dataType);
% 
%     outCSV = fullfile(outputFolder, [baseName '.csv']);
%     outMAT = fullfile(outputFolder, [baseName '.mat']);
% 
%     T = array2table(combinedData, 'VariableNames', headerCell{1});
%     writetable(T, outCSV);
%     save(outMAT, 'combinedData', 'segmentStarts', 'segmentEnds');
% 
%     %--------------------------------------------------
%     % 6) Plot (eps' ve tanδ)
%     %--------------------------------------------------
%     plotAveragedData(combinedData, outputFolder, baseName, cutoffFreqsHz, suffix, segmentStarts, segmentEnds);
% end
% 
% % =====================================================================
% % Averaged data plotting:
% %   - averagedData: [freq, eps_mean, eps_std, tan_mean, tan_std, ...]
% % =====================================================================
% function plotAveragedData(averagedData, outputFolder, fileNameOut, cutoffFreqsHz, suffix, segmentStarts, segmentEnds)
% 
%     if size(averagedData,2) < 5
%         warning('plotAveragedData: expected at least 5 columns (freq, eps_mean, eps_std, tan_mean, tan_std). Skipping plots.');
%         return;
%     end
% 
%     freqHz = averagedData(:,1);
% 
%     % eps' için üst/alt band
%     epsMean = averagedData(:,2);
%     epsStd  = averagedData(:,3);
%     upper1  = epsMean + epsStd;
%     lower1  = epsMean - epsStd;
% 
%     % tanδ için üst/alt band
%     tanMean = averagedData(:,4);
%     tanStd  = averagedData(:,5);
%     upper2  = tanMean + tanStd;
%     lower2  = tanMean - tanStd;
% 
%     % eps' shaded plot
%     plotWithShadedError(freqHz, epsMean, upper1, lower1, ...
%         segmentStarts, segmentEnds, 'b', '$\varepsilon_r^\prime$', ...
%         fullfile(outputFolder, [fileNameOut '_eps_real_shaded' suffix]));
% 
%     % tanδ shaded plot
%     plotWithShadedError(freqHz, tanMean, upper2, lower2, ...
%         segmentStarts, segmentEnds, 'r', '$tan\delta$', ...
%         fullfile(outputFolder, [fileNameOut '_tan_delta_shaded' suffix]));
% end
% 
% % =====================================================================
% % Shaded error plotting helper
% % =====================================================================
% function plotWithShadedError(freqHz, y, upper, lower, segStart, segEnd, colorChar, yLabel, savePath)
%     FontName = 'Times New Roman';
%     FontSize = 8;
% 
%     figure('Units', 'centimeters', 'Position', [1,1,18,10]); hold on;
%     xGHz = freqHz / 1e9;
% 
%     for t = 1:numel(segStart)
%         idx = segStart(t):segEnd(t);
%         if isempty(idx), continue; end
% 
%         fill([xGHz(idx); flipud(xGHz(idx))], [upper(idx); flipud(lower(idx))], ...
%             colorChar, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
%         plot(xGHz(idx), y(idx), colorChar, 'LineWidth', 2);
%     end
% 
%     grid on;
%     axis tight;
%     set(gca, 'FontName', FontName, 'FontSize', FontSize);
% 
%     xlabel('Frequency (GHz)', 'Interpreter', 'latex', 'FontName', FontName, 'FontSize', 14);
%     ylabel(yLabel,           'Interpreter', 'latex', 'FontName', FontName, 'FontSize', 14);
% 
%     saveas(gcf, [savePath '.png']);
%     savefig([savePath '.fig']);
%     close;
% end


function averagedDataLast()
% averagedDataLast
% Uses filtered merged data (Merged_Filtered_Revision) to build final
% averaged datasets for permittivity and S2P, and exports them into:
%
%   - AveragedLast_Revision        (permittivity)
%   - AveragedLast_Revision_S2P    (S2P)
%
% Behaviour:
%   - For Keysight_PNA_UoB: only permittivity exists. The same perm
%     outputs are written to BOTH AveragedLast_Revision and
%     AveragedLast_Revision_S2P.
%
%   - For Keysight_PNA_NPL:
%       * Permittivity averaged files
%         (_Averaged_Data_Roughness.csv / _Averaged_Data_withoutRoughness.csv)
%         → AveragedLast_Revision
%       * S2P averaged files
%         (_S2PAveraged_RoughnessS2P.csv / _S2PAveraged_withoutRoughnessS2P.csv)
%         → AveragedLast_Revision_S2P
%
% IMPORTANT CHANGE (requested):
%   - When stitching/merging the bands, the final combined data is sorted
%     by frequency in strictly ascending order (smallest frequency first).
%   - Band order lists are still used to DISCOVER which band a file belongs to:
%       UoB: 220-330, 325-500, 500-750, MergedBands
%       NPL: WR15, WR10, WR5, WM-380, MergedBands
%
% Old behaviour preserved:
%   - Gap-based segmentation (>2 GHz) for plotting segments
%   - CSV + MAT export
%   - Shaded-error plots for eps' and tanδ

    %------------------------------------------------------
    % 1) Root folder selection
    %------------------------------------------------------
    rootDir = uigetdir([], 'Select your root folder (RAWDATA)');
    if rootDir == 0
        disp('Operation cancelled by user.');
        return;
    end

    % Two measurement methods
    measurementMethods = {'Keysight_PNA_NPL','Keysight_PNA_UoB'};

    % File suffixes (permittivity)
    permSuffixRough = '_Averaged_Data_Roughness.csv';
    permSuffixNo    = '_Averaged_Data_withoutRoughness.csv';

    % File suffixes (S2P averaged)
    s2pSuffixRough  = '_S2PAveraged_RoughnessS2P.csv';
    s2pSuffixNo     = '_S2PAveraged_withoutRoughnessS2P.csv';

    % Sample folders
    sampleFolders = dir(rootDir);
    sampleFolders = sampleFolders([sampleFolders.isdir] & ...
                              ~startsWith({sampleFolders.name}, '.'));



    for s = 1:numel(sampleFolders)
        sampleName       = sampleFolders(s).name;
        sampleFolderPath = fullfile(rootDir, sampleName);

        for m = 1:numel(measurementMethods)
            methodName       = measurementMethods{m};
            methodFolderPath = fullfile(sampleFolderPath, methodName);

            if ~exist(methodFolderPath,'dir')
                continue;
            end

            %--------------------------------------------------
            % 2) Band names (used only for identifying a band from file name)
            %--------------------------------------------------
            if strcmp(methodName,'Keysight_PNA_UoB')
                freqBandsOrdered = {'MergedBands', '220-330','325-500','500-750'};
                cutoffFreqsHz    = [325e9, 500e9, 750e9];
            else
                freqBandsOrdered = {'MergedBands', 'WR15','WR10','WR5','WM-380'};
                cutoffFreqsHz    = [75e9, 110e9, 220e9, 500e9];
            end

            % Output folders
            outPerm = fullfile(methodFolderPath,'AveragedLast_Revision_Filtered');
            outS2P  = fullfile(methodFolderPath,'AveragedLast_Revision_S2P_Filtered');
            if ~exist(outPerm,'dir'), mkdir(outPerm); end
            if ~exist(outS2P,'dir'),  mkdir(outS2P);  end

            %--------------------------------------------------
            % 3) Scan band folders under the method folder
            %--------------------------------------------------
            bandFolders = dir(methodFolderPath);
            bandFolders = bandFolders([bandFolders.isdir] & ...
                          ~startsWith({bandFolders.name}, '.'));


            % Collected file lists
            permRoughFiles = [];
            permNoFiles    = [];
            s2pRoughFiles  = [];
            s2pNoFiles     = [];

            for b = 1:numel(bandFolders)
                bandName = bandFolders(b).name;

                % Only process known band folders
                if ~ismember(bandName, freqBandsOrdered)
                    continue;
                end

                bandFolderPath = fullfile(methodFolderPath, bandName);

                % Source folder: Merged_Filtered_Revision
                srcDir = fullfile(bandFolderPath, 'Merged_Filtered_Revision');
                if ~exist(srcDir, 'dir')
                    continue;
                end

                csvFiles = dir(fullfile(srcDir, '*.csv'));
                if isempty(csvFiles)
                    continue;
                end

                for k = 1:numel(csvFiles)
                    fname = csvFiles(k).name;
                    csvFiles(k).folder = srcDir;

                    if endsWith(fname, permSuffixRough, 'IgnoreCase', true)
                        permRoughFiles = [permRoughFiles; csvFiles(k)]; %#ok<AGROW>
                    elseif endsWith(fname, permSuffixNo, 'IgnoreCase', true)
                        permNoFiles    = [permNoFiles; csvFiles(k)]; %#ok<AGROW>
                    elseif endsWith(fname, s2pSuffixRough, 'IgnoreCase', true)
                        s2pRoughFiles  = [s2pRoughFiles; csvFiles(k)]; %#ok<AGROW>
                    elseif endsWith(fname, s2pSuffixNo, 'IgnoreCase', true)
                        s2pNoFiles     = [s2pNoFiles; csvFiles(k)]; %#ok<AGROW>
                    end
                end
            end

            %--------------------------------------------------
            % 4) PERMITTIVITY (NPL + UoB)
            %--------------------------------------------------
            if ~isempty(permRoughFiles)
                fprintf('\n📘 PERM Roughness: %s | %s\n', sampleName, methodName);
                processAndSaveAveraged(permRoughFiles, freqBandsOrdered, ...
                    outPerm, sampleName, 'Averaged_Roughness', cutoffFreqsHz, '_roughness');

                % UoB: also write same perm outputs into S2P folder
                if strcmp(methodName,'Keysight_PNA_UoB')
                    processAndSaveAveraged(permRoughFiles, freqBandsOrdered, ...
                        outS2P, sampleName, 'Averaged_Roughness', cutoffFreqsHz, '_roughness');
                end
            end

            if ~isempty(permNoFiles)
                fprintf('\n📘 PERM WithoutRough: %s | %s\n', sampleName, methodName);
                processAndSaveAveraged(permNoFiles, freqBandsOrdered, ...
                    outPerm, sampleName, 'Averaged', cutoffFreqsHz, '');

                if strcmp(methodName,'Keysight_PNA_UoB')
                    processAndSaveAveraged(permNoFiles, freqBandsOrdered, ...
                        outS2P, sampleName, 'Averaged', cutoffFreqsHz, '');
                end
            end

            %--------------------------------------------------
            % 5) S2P (meaningful only for NPL)
            %--------------------------------------------------
            if ~strcmp(methodName,'Keysight_PNA_NPL')
                continue;
            end

            if ~isempty(s2pRoughFiles)
                fprintf('\n📗 S2P Roughness: %s | %s\n', sampleName, methodName);
                processAndSaveAveraged(s2pRoughFiles, freqBandsOrdered, ...
                    outS2P, sampleName, 'S2PAveraged_RoughnessS2P', cutoffFreqsHz, '_roughnessS2P');
            end

            if ~isempty(s2pNoFiles)
                fprintf('\n📗 S2P WithoutRough: %s | %s\n', sampleName, methodName);
                processAndSaveAveraged(s2pNoFiles, freqBandsOrdered, ...
                    outS2P, sampleName, 'S2PAveraged_withoutRoughnessS2P', cutoffFreqsHz, '_withoutRoughnessS2P');
            end

        end
    end

    fprintf('\n✅ All samples / methods processed.\n');
end


% =====================================================================
% Process and save averaged files:
%  - fileList          : list of averaged CSVs to stitch
%  - freqBandsOrdered  : band name list (used to identify band from filename)
%  - outputFolder      : output folder
%  - sampleName        : sample name
%  - dataType          : label for output file
%  - cutoffFreqsHz     : used in plots (kept)
%  - suffix            : extra suffix for plot filenames
%
% IMPORTANT CHANGE:
%  - After concatenation, combinedData is sorted by frequency ascending.
%  - Segment detection for plotting is then computed AFTER sorting.
% =====================================================================
function processAndSaveAveraged(fileList, freqBandsOrdered, outputFolder, sampleName, dataType, cutoffFreqsHz, suffix)
    if isempty(fileList)
        return;
    end

    fprintf('   → Processing %s (%d files)\n', dataType, numel(fileList));

    dataCell   = cell(numel(fileList), 1);
    headerCell = cell(numel(fileList), 1);
    bandIndex  = zeros(numel(fileList), 1);

    %--------------------------------------------------
    % 1) Read files and determine band index from filename
    %--------------------------------------------------
    for j = 1:numel(fileList)
        fPath    = fullfile(fileList(j).folder, fileList(j).name);
        cellData = readcell(fPath);

        headerCell{j} = cellData(1,:);
        dataOnly      = cellData(2:end,:);

        numericMatrix = cell2mat(cellfun(@double, dataOnly, 'UniformOutput', false));

        % Ensure each file itself is ordered by frequency
        if ~isempty(numericMatrix) && size(numericMatrix,2) >= 1
            numericMatrix = sortrows(numericMatrix, 1, 'ascend');
        end
        dataCell{j} = numericMatrix;

        % Robust band detection from filename:
        % - finds the first matching band token in freqBandsOrdered
        bandIndex(j) = getBandIndexFromFilename(fileList(j).name, freqBandsOrdered);
    end

    %--------------------------------------------------
    % 2) Sort by bandIndex (only for stable stitching source order)
    %--------------------------------------------------
    [~, sortIdx]   = sort(bandIndex);
    sortedDataCell = dataCell(sortIdx);
    headerCell     = headerCell(sortIdx);

    %--------------------------------------------------
    % 3) Concatenate all data
    %--------------------------------------------------
    combinedData = vertcat(sortedDataCell{:});
    if isempty(combinedData)
        return;
    end

    %--------------------------------------------------
    % 4) IMPORTANT: sort the final combined data by frequency ascending
    %--------------------------------------------------
    combinedData = sortrows(combinedData, 1, 'ascend');

    %--------------------------------------------------
    % 5) Segment detection after sorting (gap > 2 GHz)
    %--------------------------------------------------
    totalRows = size(combinedData,1);
    if totalRows <= 1
        segmentStarts = 1;
        segmentEnds   = totalRows;
    else
        freqDiff   = diff(combinedData(:,1));
        gapPoints  = find(freqDiff > 2e9);

        segmentStarts = [1; gapPoints + 1];
        segmentEnds   = [gapPoints; totalRows];
    end

    %--------------------------------------------------
    % 6) Save CSV + MAT
    %--------------------------------------------------
    fileNameOut = regexprep(sampleName, '\s+', '_');
    baseName    = sprintf('%s_%s', fileNameOut, dataType);

    outCSV = fullfile(outputFolder, [baseName '.csv']);
    outMAT = fullfile(outputFolder, [baseName '.mat']);

    T = array2table(combinedData, 'VariableNames', headerCell{1});
    writetable(T, outCSV);
    save(outMAT, 'combinedData', 'segmentStarts', 'segmentEnds');

    %--------------------------------------------------
    % 7) Plot (eps' and tanδ)
    %--------------------------------------------------
    plotAveragedData(combinedData, outputFolder, baseName, cutoffFreqsHz, suffix, segmentStarts, segmentEnds);
end


% =====================================================================
% Band index detection from filename (robust)
%   - Returns the index of the first band name found in the filename.
%   - If none found, returns Inf (sorted last).
% =====================================================================
function idx = getBandIndexFromFilename(fname, freqBandsOrdered)
    fnameLow = lower(string(fname));

    idx = Inf;
    for k = 1:numel(freqBandsOrdered)
        token = lower(string(freqBandsOrdered{k}));

        % Match either as a substring or as an underscore-separated token
        if contains(fnameLow, token)
            idx = k;
            return;
        end
    end
end


% =====================================================================
% Averaged data plotting:
%   - averagedData: [freq, eps_mean, eps_std, tan_mean, tan_std, ...]
% =====================================================================
function plotAveragedData(averagedData, outputFolder, fileNameOut, cutoffFreqsHz, suffix, segmentStarts, segmentEnds)

    if size(averagedData,2) < 5
        warning('plotAveragedData: expected at least 5 columns (freq, eps_mean, eps_std, tan_mean, tan_std). Skipping plots.');
        return;
    end

    freqHz = averagedData(:,1);

    epsMean = averagedData(:,2);
    epsStd  = averagedData(:,3);
    upper1  = epsMean + epsStd;
    lower1  = epsMean - epsStd;

    tanMean = averagedData(:,4);
    tanStd  = averagedData(:,5);
    upper2  = tanMean + tanStd;
    lower2  = tanMean - tanStd;

    plotWithShadedError(freqHz, epsMean, upper1, lower1, ...
        segmentStarts, segmentEnds, 'b', '$\varepsilon_r^\prime$', ...
        fullfile(outputFolder, [fileNameOut '_eps_real_shaded' suffix]), cutoffFreqsHz);

    plotWithShadedError(freqHz, tanMean, upper2, lower2, ...
        segmentStarts, segmentEnds, 'r', '$tan\delta$', ...
        fullfile(outputFolder, [fileNameOut '_tan_delta_shaded' suffix]), cutoffFreqsHz);
end


% =====================================================================
% Shaded error plotting helper
% =====================================================================
function plotWithShadedError(freqHz, y, upper, lower, segStart, segEnd, colorChar, yLabel, savePath, cutoffFreqsHz)
    FontName = 'Times New Roman';
    FontSize = 8;

    figure('Units', 'centimeters', 'Position', [1,1,18,10]); hold on;
    xGHz = freqHz / 1e9;

    for t = 1:numel(segStart)
        idx = segStart(t):segEnd(t);
        if isempty(idx), continue; end

        fill([xGHz(idx); flipud(xGHz(idx))], [upper(idx); flipud(lower(idx))], ...
            colorChar, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        plot(xGHz(idx), y(idx), colorChar, 'LineWidth', 2);
    end

    % Optional cutoff markers (kept; can remove if not needed)
    if nargin >= 10 && ~isempty(cutoffFreqsHz)
        for c = 1:numel(cutoffFreqsHz)
            xline(cutoffFreqsHz(c)/1e9, '--k', 'LineWidth', 1);
        end
    end

    grid on;
    axis tight;
    set(gca, 'FontName', FontName, 'FontSize', FontSize);

    xlabel('Frequency (GHz)', 'Interpreter', 'latex', 'FontName', FontName, 'FontSize', 14);
    ylabel(yLabel,           'Interpreter', 'latex', 'FontName', FontName, 'FontSize', 14);

    saveas(gcf, [savePath '.png']);
    savefig([savePath '.fig']);
    close;
end
