function folderRemover()
    answer = questdlg('Do you want to delete folders?', 'Delete Confirmation', 'Yes','No','No');
    if ~strcmp(answer, 'Yes'), return; end

    rootDir = uigetdir([], 'Select the root directory (folders named "pulsefft" will be deleted)');
    if rootDir == 0; return; end

    % Get name of the folder
    prompt = {'Enter the name of the folders to delete:'};
    dlgtitle = 'Folder Name Input';
    dims = [1 50];
    definput = {'Merged'};
    folderNameInput = inputdlg(prompt, dlgtitle, dims, definput);

    if isempty(folderNameInput), return; end
    folderName = strtrim(folderNameInput{1});

    % Scannin requested folders
    matchedDirs = dir(fullfile(rootDir, '**', folderName));
    matchedDirs = matchedDirs([matchedDirs.isdir]);

    % 5. filter those ".."
    validDirs = matchedDirs(~strcmp({matchedDirs.name}, '..'));

    if isempty(validDirs)
        msgbox('No folders found matching that name.','Info');
        return;
    end

    relativePaths = strrep(arrayfun(@(d) fullfile(d.folder, d.name), validDirs, 'UniformOutput', false), [rootDir filesep], '');
    folderList = join(relativePaths, newline);
    confirmMsg = sprintf('The following folders will be deleted:\n\n%s\n\nAre you sure?', folderList{:});
    confirmAnswer = questdlg(confirmMsg, 'Final Confirmation', 'Yes','No','No');

    if ~strcmp(confirmAnswer, 'Yes'), return; end

    % Proceed with deletion
    for i = 1:length(validDirs)
        mergedPath = fullfile(validDirs(i).folder, validDirs(i).name);
        fprintf('Deleting: %s\n', mergedPath);

        try
            rmdir(mergedPath, 's');
        catch ME
            fprintf('! Error: Could not delete %s (%s)\n', mergedPath, ME.message);
        end
    end

    msgbox('✅ Selected folders have been deleted safely.', 'Success');
end