function fileRenamer()
    % Create a simple GUI for selecting mode and starting renaming
    fig = uifigure('Name', 'File Renamer', 'Position', [100 100 400 200]);
    uilabel(fig, 'Position', [30 140 100 22], 'Text', 'Select Mode:');
    ddMode = uidropdown(fig, 'Items', {'TDS', 'PNA', 'All'}, 'Position', [130 140 150 22]);
    uibutton(fig, 'Text', 'Run Renaming', 'Position', [130 80 150 30], 'ButtonPushedFcn', @(btn, event) onRun(ddMode.Value, fig));
end

function onRun(mode, fig)
    close(fig);
    runRenaming(mode);
end

function runRenaming(mode)
% This code rename the files according to selection during the naming keep
% name of the images. 
% If there is a another files Film and Plate inside pulse or spectr folders 
% include them.
% In TDS reference data named all of them as a reference.txt if there is
    try
        rootDir = uigetdir([], 'Select the main directory');
        if rootDir == 0, return; end
        mode = lower(mode);

        % Writing and adding mode
        logFile = fullfile(rootDir, 'rename_log.txt');
        if strcmp(mode, 'all') || ~isfile(logFile)
            fid = fopen(logFile, 'w');
            fprintf(fid, '--- File Renaming Log ---\n\n');
        else
            fid = fopen(logFile, 'a');
            fprintf(fid, '\n--- New Operation (%s) ---\n\n', upper(mode));
        end

        if strcmp(mode, 'tds') || strcmp(mode, 'all')
            renameTDS(rootDir, fid);
        end
        if strcmp(mode, 'pna') || strcmp(mode, 'all')
            renamePNA(rootDir, fid);
        end

        fclose(fid);

        msg = sprintf('Renaming complete.%sLog saved to:%s%s', newline, newline, logFile);
        f = uifigure('Name', 'Notification', 'Position', [100 100 400 100]);
        uialert(f, msg, 'Done', 'Icon', 'success');

    catch ME
        disp(getReport(ME));
        f = uifigure('Name', 'Error', 'Position', [100 100 400 100]);
        uialert(f, ['Error: ' ME.message], 'Error', 'Icon', 'error');
    end
end


function renameTDS(rootDir, fid)
    imageExts = {'.jpg', '.jpeg', '.png', '.bmp'};
    targetExt = '.txt';
    isTarget = @(name) any(strcmp(name, {'pulse', 'spectr'}));

    allDirs = dir(fullfile(rootDir, '**'));
    allDirs = allDirs([allDirs.isdir]);
    targetFolders = allDirs(arrayfun(@(d) isTarget(d.name), allDirs));

    for i = 1:numel(targetFolders)
        baseFolder = fullfile(targetFolders(i).folder, targetFolders(i).name);
        subdirs = dir(baseFolder);
        subdirs = subdirs([subdirs.isdir] & ~startsWith({subdirs.name}, '.'));

        if any(strcmp({subdirs.name}, 'Film')) || any(strcmp({subdirs.name}, 'Plate'))
            subfolders = subdirs(strcmp({subdirs.name}, 'Film') | strcmp({subdirs.name}, 'Plate'));
        else
            subfolders = struct('name', {''}, 'folder', baseFolder);
        end

        for k = 1:numel(subfolders)
            if isfield(subfolders(k), 'folder')
                currentPath = fullfile(subfolders(k).folder, subfolders(k).name);
                filmOrPlate = subfolders(k).name;
            else
                currentPath = baseFolder;
                filmOrPlate = '';
            end
            files = dir(fullfile(currentPath, '*'));
            files = files(~[files.isdir]);

            if isempty(files)
                fprintf(fid, '[%s] No files to process.\n', currentPath); continue;
            end

            mainName = getMainName(currentPath);
            if isempty(mainName)
                fprintf(fid, '[%s] Main folder name not found. Skipped.\n', currentPath); continue;
            end

            fileCount = 1;
            for j = 1:numel(files)
                oldName = files(j).name;
                [~, ~, ext] = fileparts(oldName);
                oldFullPath = fullfile(currentPath, oldName);

                if any(strcmpi(ext, imageExts))
                    fprintf(fid, '• Skipped image file: %s\n', oldName); continue;
                end

                if contains(lower(oldName), 'reference') || contains(lower(oldName), 'ref2')
                    newName = 'reference.txt';
                else
                    if ~isempty(filmOrPlate)
                        newName = sprintf('%s_%s_%d%s', mainName, filmOrPlate, fileCount, targetExt);
                    else
                        newName = sprintf('%s_%d%s', mainName, fileCount, targetExt);
                    end
                    fileCount = fileCount + 1;
                end

                newFullPath = fullfile(currentPath, newName);
                if ~strcmp(oldFullPath, newFullPath)
                    movefile(oldFullPath, newFullPath);
                    fprintf(fid, '✓ %s -> %s\n', oldName, newName);
                else
                    fprintf(fid, '• Already correct: %s\n', oldName);
                end
            end
            fprintf(fid, '\n');
        end
    end
end

function renamePNA(rootDir, fid)
    keysightFolders = {'Keysight_PNA_NPL', 'Keysight_PNA_UoB'};
    validExts = {'.txt', '.csv', '.xlsx'};

    for k = 1:numel(keysightFolders)
        matchDirs = dir(fullfile(rootDir, '**', keysightFolders{k}));
        matchDirs = matchDirs([matchDirs.isdir]);

        for m = 1:numel(matchDirs)
            targetPath = fullfile(matchDirs(m).folder, matchDirs(m).name);
            
            % Skip if directory is Merged or AveragedLast
            if any(contains(targetPath, {'Merged', 'AveragedLast'}, 'IgnoreCase', true))
                fprintf(fid, '[%s] Skipped (excluded folder).\n', targetPath);
                continue;
            end
            
            files = dir(fullfile(targetPath, '*'));
            files = files(~[files.isdir]);

            if isempty(files)
                fprintf(fid, '[%s] No files found.\n', targetPath); continue;
            end

            mainName = getMainName(targetPath);
            if isempty(mainName)
                fprintf(fid, '[%s] Main folder name not found. Skipped.\n', targetPath); continue;
            end

            fileCount = 1;
            for j = 1:numel(files)
                [~, ~, ext] = fileparts(files(j).name);
                if ~any(strcmpi(ext, validExts))
                    fprintf(fid, '• Skipped non-target file: %s\n', files(j).name); continue;
                end

                newName = sprintf('%s_%d%s', mainName, fileCount, ext);
                oldFullPath = fullfile(targetPath, files(j).name);
                newFullPath = fullfile(targetPath, newName);
                fileCount = fileCount + 1;

                if ~strcmp(oldFullPath, newFullPath)
                    movefile(oldFullPath, newFullPath);
                    fprintf(fid, '✓ %s -> %s\n', files(j).name, newName);
                else
                    fprintf(fid, '• Already correct: %s\n', files(j).name);
                end
            end
            fprintf(fid, '\n');
        end
    end
end

function mainName = getMainName(path)
    parts = split(path, filesep);
    idx = find(contains(parts, 'ROGERS_') | contains(parts, 'RADIX_'), 1, 'last');
    if isempty(idx)
        mainName = '';
    elseif idx < length(parts)
        mainName = sprintf('%s_%s', parts{idx}, parts{idx + 1});
    else
        mainName = parts{idx};
    end
end
    
 