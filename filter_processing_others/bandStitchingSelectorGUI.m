function selections = bandStitchingSelectorGUI()
% bandStitchingSelectorGUI (SINGLE FILE, classic figure)
%
% Behavior (old-style, no manual "source folders"):
% - User selects rootDir, then Sample + Method (e.g., UoB / NPL)
% - GUI auto-discovers band folders under <root>/<sample>/<methodFolder>/<band>
%   (also supports <band>/Merged_Filtered_Revision automatically)
% - User multi-selects bands to stitch
% - For each selected band, user picks exactly ONE FileId (ID=last number in filename)
% - Default per band: FileId=1 if exists, else smallest available
% - Preview shows stitched permittivity + S2P (if defined in JSON)
% - Add Set locks the combination and prevents re-using the same FileId in that band
% - Generate writes outputs under:
%     <root>/<sample>/<methodFolder>/MergedBands/<Band1_Band2_...>/
%   and writes a log under rootDir.
%
% NEW (ONLY TWO BUTTONS ADDED):
% - Two checkboxes (default ON):
%     [x] Save CSV (Permittivity): writes *_Merged_Data.csv + *_Averaged_Data.csv
%     [x] Save S2P:               writes *_S2PMerged.csv  + *_S2PAveraged_Data.csv
%
% Output behavior (old-style merged):
% - For each group (same Sample/Method + same band-combo folder):
%   * MERGED files contain ALL added sets (side-by-side columns).
%   * AVERAGED files average across those sets per frequency (perm: mean+std; S2P: complex mean).
%
% Config: config_VNA.json must contain:
%   config.MergedProcessing.Methods(...)
%   Each method may have:
%     - FolderName, Name
%     - Bands: Name, LowerCutoffHz, UpperCutoffHz
%     - PermittivityInput: Extensions, FrequencyColumn, EpsRealColumn, TanColumn, ...
%     - SParameterInput: Extensions, Columns, (optional) DataFormat/FreqUnit etc.
%
% Author: adapted to requested workflow

    % ---------------- CONFIG ----------------
    configPath = 'config_VNA.json';
    if ~isfile(configPath)
        error('Config file not found: %s', configPath);
    end
    cfgText = fileread(configPath);
    config  = jsondecode(cfgText);

    if ~isfield(config,'MergedProcessing') || ~isfield(config.MergedProcessing,'Methods')
        error('config.MergedProcessing.Methods missing.');
    end
    mp      = config.MergedProcessing;
    methods = mp.Methods;

    % Root dir from config if exists, else uigetdir
    rootDir = '';
    if isfield(config,'FilePatterns') && isfield(config.FilePatterns,'rootDir')
        rootDir = config.FilePatterns.rootDir;
        if isstring(rootDir), rootDir = char(rootDir); end
    end
    if isempty(rootDir) || ~isfolder(rootDir)
        rootDir = uigetdir(pwd,'Select rootDir (contains sample folders)');
        if isequal(rootDir,0), selections = []; return; end
    end

    % Sample folders
    dd = dir(rootDir);
    dd = dd([dd.isdir] & ~startsWith({dd.name}, '.'));
    sampleNames = {dd.name};
    if isempty(sampleNames)
        error('No sample folders under: %s', rootDir);
    end

    % Method names for popup
    methodNames = cell(numel(methods),1);
    for i=1:numel(methods)
        methodNames{i} = char(methods(i).Name);
    end

    % ---------------- STATE ----------------
    state.rootDir      = rootDir;
    state.sampleNames  = sampleNames;
    state.methods      = methods;

    % used IDs per band (per sample+method session)
    % key: bandName, value: numeric vector used IDs
    state.usedIdsMap   = containers.Map('KeyType','char','ValueType','any');

    % current per-band selectors
    state.selBandsSorted = {};
    state.bandSelectors  = struct('band',{},'popup',{},'text',{},'ids',[],'names',{{}});

    % added sets (locked combos)
    % each row: SampleRel, MethodName, Bands, FileIds, PermFileNames, S2PFileNames, CutoffsHz
    combos = struct('SampleRel',{},'MethodName',{},'MethodFolderName',{}, ...
                    'BandNames',{},'CutoffsHz',{}, ...
                    'FileIds',{},'PermFileNames',{},'S2PFileNames',{}, ...
                    'OutFolder',{});

    selectedRow = [];

    % ---------------- FIGURE LAYOUT ----------------
    fig = figure('Name','Band Stitching Selector (per-band FileId, no manual sources)', ...
        'NumberTitle','off','MenuBar','none','ToolBar','none', 'Position',[80 80 1320 760]);

    pnlL = uipanel('Parent',fig,'Title','Selection','Units','pixels','Position',[10 10 420 740]);

    uicontrol('Parent',pnlL,'Style','text','Units','pixels','Position',[10 705 400 22], ...
        'HorizontalAlignment','left','String',['Root: ' rootDir]);

    uicontrol('Parent',pnlL,'Style','text','Units','pixels','Position',[10 675 80 18], ...
        'HorizontalAlignment','left','String','Sample:');
    popSample = uicontrol('Parent',pnlL,'Style','popupmenu','Units','pixels','Position',[90 675 320 22], ...
        'String',sampleNames,'Value',1,'Callback',@refreshBandsAndSelectors);

    uicontrol('Parent',pnlL,'Style','text','Units','pixels','Position',[10 645 80 18], ...
        'HorizontalAlignment','left','String','Method:');
    popMethod = uicontrol('Parent',pnlL,'Style','popupmenu','Units','pixels','Position',[90 645 320 22], ...
        'String',methodNames,'Value',1,'Callback',@refreshBandsAndSelectors);

    uicontrol('Parent',pnlL,'Style','text','Units','pixels','Position',[10 615 400 18], ...
        'HorizontalAlignment','left', ...
        'String','Bands (multi-select). Band order is auto by JSON LowerCutoffHz:');

    lstBands = uicontrol('Parent',pnlL,'Style','listbox','Units','pixels','Position',[10 420 400 200], ...
        'String',{},'Max',30,'Min',0,'Callback',@onBandsChanged);

    % dynamic panel for per-band id pickers
    pnlPick = uipanel('Parent',pnlL,'Title','Per-band FileId selection (each band used once)','Units','pixels', ...
        'Position',[10 230 400 180]);

    % === ONLY TWO NEW "BUTTONS" (checkboxes), default ON ===
    chkSavePerm = uicontrol('Parent',pnlPick,'Style','checkbox','Units','pixels', ...
        'Position',[10 8 185 18], 'String','Save CSV (Permittivity)', 'Value',1);
    chkSaveS2P  = uicontrol('Parent',pnlPick,'Style','checkbox','Units','pixels', ...
        'Position',[210 8 180 18], 'String','Save S2P', 'Value',1);

    btnPreview = uicontrol('Parent',pnlL,'Style','pushbutton','Units','pixels','Position',[10 195 400 28], ...
        'String','Preview stitched data (current FileIds)','Callback',@onPreview);

    btnAdd = uicontrol('Parent',pnlL,'Style','pushbutton','Units','pixels','Position',[10 160 400 28], ...
        'String','Add Set (lock this combination, prevent re-use)','Callback',@onAddSet);

    tbl = uitable('Parent',pnlL,'Units','pixels','Position',[10 40 400 115], ...
        'ColumnName',{'Sample/Method','Bands','FileIds'}, ...
        'Data',cell(0,3), ...
        'CellSelectionCallback',@onTableSelect);

    btnRemove = uicontrol('Parent',pnlL,'Style','pushbutton','Units','pixels','Position',[10 10 195 24], ...
        'String','Remove selected set','Callback',@onRemoveSet);

    btnGenerate = uicontrol('Parent',pnlL,'Style','pushbutton','Units','pixels','Position',[215 10 195 24], ...
        'String','Generate + Log + Close','Callback',@onGenerateClose);

    % Right panel: preview axes
    pnlR = uipanel('Parent',fig,'Title','Preview (stitched)','Units','pixels','Position',[440 10 870 740]);
    axEps   = axes('Parent',pnlR,'Units','pixels','Position',[40 420 380 280]);
    axTan   = axes('Parent',pnlR,'Units','pixels','Position',[40 90  380 280]);
    axMag   = axes('Parent',pnlR,'Units','pixels','Position',[460 420 380 280]);
    axPhase = axes('Parent',pnlR,'Units','pixels','Position',[460 90  380 280]);

    refreshBandsAndSelectors();
    selections = [];
    uiwait(fig);

    % ===================== CALLBACKS =====================

    function onTableSelect(~,evt)
        if isempty(evt.Indices)
            selectedRow = [];
        else
            selectedRow = evt.Indices(1);
        end
    end

    function refreshBandsAndSelectors(~,~)
        refreshBands();
        rebuildBandSelectors();
        onPreview();
    end

    function refreshBands()
        mIdx = get(popMethod,'Value');
        sIdx = get(popSample,'Value');
        method = state.methods(mIdx);

        samplePath = fullfile(state.rootDir, state.sampleNames{sIdx});
        methodPath = fullfile(samplePath, char(method.FolderName));

        if ~isfolder(methodPath)
            set(lstBands,'String',{},'Value',[]);
            return;
        end

        bandNames = {method.Bands.Name};
        existMask = false(size(bandNames));
        for iB=1:numel(bandNames)
            existMask(iB) = isfolder(fullfile(methodPath, char(bandNames{iB})));
        end
        bandNames = bandNames(existMask);

        set(lstBands,'String',bandNames);
        if ~isempty(bandNames)
            set(lstBands,'Value',1);
        else
            set(lstBands,'Value',[]);
        end
    end

    function onBandsChanged(~,~)
        rebuildBandSelectors();
        onPreview();
    end

    function rebuildBandSelectors()
        % clear old controls inside pnlPick but keep the two checkboxes
        kids = allchild(pnlPick);
        for kk=1:numel(kids)
            if kids(kk) ~= chkSavePerm && kids(kk) ~= chkSaveS2P
                delete(kids(kk));
            end
        end

        state.bandSelectors = struct('band',{},'popup',{},'text',{},'ids',[],'names',{{}});
        state.selBandsSorted = {};

        bandList = get(lstBands,'String');
        selIdx   = get(lstBands,'Value');
        if isempty(bandList) || isempty(selIdx)
            return;
        end

        % band selection sorted by JSON LowerCutoffHz
        mIdx = get(popMethod,'Value');
        method = state.methods(mIdx);

        selectedBands = bandList(selIdx);
        cut = zeros(numel(selectedBands),2);
        for i=1:numel(selectedBands)
            cut(i,:) = resolveBandCutoffHz_jsonOnly(method, selectedBands{i});
        end
        [~,ord] = sort(cut(:,1),'ascend');
        selectedBands = selectedBands(ord);
        state.selBandsSorted = selectedBands;

        % Build per-band popup rows
        topY = 140;     % leave space for checkbox row at bottom
        rowH = 30;

        for iB = 1:numel(selectedBands)
            band = selectedBands{iB};

            uicontrol('Parent',pnlPick,'Style','text','Units','pixels', ...
                'Position',[10 topY-(iB-1)*rowH 130 18], ...
                'HorizontalAlignment','left', ...
                'String',sprintf('Band %d: %s', iB, band));

            % list available files (perm preferred; if perm missing, fall back to s2p)
            [ids, names] = listBandFilesAsIds(band);

            % Apply "used once per band" constraint
            used = getUsedIdsForBand(band);
            keepMask = ~ismember(ids, used);
            idsAvail = ids(keepMask);
            namesAvail = names(keepMask);

            if isempty(idsAvail)
                popupStr = {'(no IDs left)'};
                popupVal = 1;
            else
                popupStr = makeIdDisplayStrings(idsAvail, namesAvail);
                % default = ID=1 if present else smallest
                [popupVal, ~] = defaultPopupIndex(idsAvail);
            end

            pop = uicontrol('Parent',pnlPick,'Style','popupmenu','Units','pixels', ...
                'Position',[150 topY-(iB-1)*rowH 160 22], ...
                'String',popupStr,'Value',popupVal, ...
                'Callback',@(src,evt)onBandIdChanged());

            txt = uicontrol('Parent',pnlPick,'Style','text','Units','pixels', ...
                'Position',[315 topY-(iB-1)*rowH 80 18], ...
                'HorizontalAlignment','left', 'String','');

            state.bandSelectors(iB).band  = band;
            state.bandSelectors(iB).popup = pop;
            state.bandSelectors(iB).text  = txt;
            state.bandSelectors(iB).ids   = idsAvail;
            state.bandSelectors(iB).names = namesAvail;

            updateFilenameTextForRow(iB);
        end
    end

    function onBandIdChanged()
        for iB=1:numel(state.bandSelectors)
            updateFilenameTextForRow(iB);
        end
        onPreview();
    end

    function updateFilenameTextForRow(iB)
        if iB > numel(state.bandSelectors), return; end
        row = state.bandSelectors(iB);
        if isempty(row.ids)
            set(row.text,'String','');
            return;
        end
        v = get(row.popup,'Value');
        v = max(1, min(v, numel(row.ids)));
        fname = row.names{v};
        if numel(fname) > 10
            short = [fname(1:8) '...'];
        else
            short = fname;
        end
        set(row.text,'String',short);
    end

    function onPreview(~,~)
        cla(axEps); cla(axTan); cla(axMag); cla(axPhase);

        if isempty(state.selBandsSorted) || isempty(state.bandSelectors)
            axes(axEps); title(axEps,'Select bands to preview');
            return;
        end

        [fileIds, permFileNames, s2pFileNames, cutHz, methodPathAbs, method] = getCurrentSelection();
        if isempty(methodPathAbs)
            axes(axEps); title(axEps,'Invalid sample/method path');
            return;
        end

        selLine = selectionSummaryLine(state.selBandsSorted, fileIds, permFileNames);

        % ---- Permittivity stitched ----
        if isfield(method,'PermittivityInput')
            permCfg = method.PermittivityInput;
            dperm = stitchPermFromFileNames(methodPathAbs, state.selBandsSorted, cutHz, permCfg, permFileNames);
            if ~isempty(dperm)
                dperm = sortrows(dperm,1);
                xGHz = dperm(:,1)/1e9;

                axes(axEps);
                plot(xGHz, dperm(:,2)); grid on;
                title(axEps, ['EpsReal stitched | ' selLine],'Interpreter','none');
                xlabel(axEps,'Frequency (GHz)'); ylabel(axEps,'EpsReal');

                axes(axTan);
                plot(xGHz, dperm(:,3)); grid on;
                title(axTan, ['TanDelta stitched | ' selLine],'Interpreter','none');
                xlabel(axTan,'Frequency (GHz)'); ylabel(axTan,'TanDelta');
            else
                axes(axEps); title(axEps,'No permittivity data for this selection');
            end
        else
            axes(axEps); title(axEps,'PermittivityInput not defined in JSON');
        end

        % ---- S2P stitched ----
        if isfield(method,'SParameterInput')
            spCfg = method.SParameterInput;
            ds2p = stitchS2PFromFileNames(methodPathAbs, state.selBandsSorted, cutHz, spCfg, s2pFileNames);
            if ~isempty(ds2p)
                ds2p = sortrows(ds2p,1);
                xGHz = ds2p(:,1)/1e9;

                axes(axMag); hold on;
                plot(xGHz, ds2p(:,2));  % S11 mag
                plot(xGHz, ds2p(:,4));  % S21 mag
                plot(xGHz, ds2p(:,6));  % S12 mag
                plot(xGHz, ds2p(:,8));  % S22 mag
                hold off; grid on;
                title(axMag, ['|Sij| stitched | ' selLine],'Interpreter','none');
                xlabel(axMag,'Frequency (GHz)'); ylabel(axMag,'Magnitude');
                legend(axMag, {'S11','S21','S12','S22'}, 'Location','best');

                ph11 = rad2deg(unwrap(deg2rad(ds2p(:,3))));
                ph21 = rad2deg(unwrap(deg2rad(ds2p(:,5))));
                ph12 = rad2deg(unwrap(deg2rad(ds2p(:,7))));
                ph22 = rad2deg(unwrap(deg2rad(ds2p(:,9))));

                axes(axPhase); hold on;
                plot(xGHz, ph11);
                plot(xGHz, ph21);
                plot(xGHz, ph12);
                plot(xGHz, ph22);
                hold off; grid on;
                title(axPhase, ['Unwrapped phase stitched | ' selLine],'Interpreter','none');
                xlabel(axPhase,'Frequency (GHz)'); ylabel(axPhase,'Phase (deg)');
                legend(axPhase, {'S11','S21','S12','S22'}, 'Location','best');
            else
                axes(axMag); title(axMag,'No S2P data for this selection');
            end
        else
            axes(axMag); title(axMag,'SParameterInput not defined in JSON (ok for UoB-perm only)');
        end
    end

    function onAddSet(~,~)
        if isempty(state.selBandsSorted) || numel(state.selBandsSorted) < 2
            errordlg('Select at least 2 bands to stitch.','Need 2+ bands');
            return;
        end

        [fileIds, permFileNames, s2pFileNames, cutHz, methodPathAbs, method] = getCurrentSelection();
        if isempty(methodPathAbs)
            errordlg('Invalid sample/method path.','Path error');
            return;
        end

        if any(isnan(fileIds)) || any(fileIds==0)
            errordlg('Some bands have no available FileId left (already used).','No IDs left');
            return;
        end

        % Lock: mark used IDs per band
        for iB=1:numel(state.selBandsSorted)
            band = state.selBandsSorted{iB};
            markUsedId(band, fileIds(iB));
        end

        % Build combo record
        sIdx = get(popSample,'Value');
        mIdx = get(popMethod,'Value');
        methodFolderName = char(state.methods(mIdx).FolderName);

        sampleRel = fullfile(state.sampleNames{sIdx}, methodFolderName);
        comboName = strjoin(state.selBandsSorted, '_');
        outFolderRel = fullfile('MergedBands', comboName);

        combos(end+1) = struct( ...
            'SampleRel', sampleRel, ...
            'MethodName', char(method.Name), ...
            'MethodFolderName', methodFolderName, ...
            'BandNames', {state.selBandsSorted}, ...
            'CutoffsHz', cutHz, ...
            'FileIds', fileIds, ...
            'PermFileNames', {permFileNames}, ...
            'S2PFileNames',  {s2pFileNames}, ...
            'OutFolder', outFolderRel );

        % Update table
        data = get(tbl,'Data');
        data(end+1,:) = {sampleRel, strjoin(state.selBandsSorted, ', '), mat2str(fileIds)};
        set(tbl,'Data',data);

        rebuildBandSelectors();
        onPreview();
    end

    function onRemoveSet(~,~)
        data = get(tbl,'Data');
        if isempty(data), return; end
        if isempty(selectedRow) || selectedRow < 1 || selectedRow > size(data,1)
            errordlg('Select a row from the table.','No row selected');
            return;
        end

        % NOTE: Removing a set will NOT free used IDs (keeps strict "used once per band" behavior).
        data(selectedRow,:) = [];
        set(tbl,'Data',data);

        if selectedRow <= numel(combos)
            combos(selectedRow) = [];
        end
        selectedRow = [];

        onPreview();
    end

    function onGenerateClose(~,~)
        if isempty(combos)
            errordlg('No sets added. Use "Add Set" first.','Nothing to generate');
            return;
        end

        doPerm = get(chkSavePerm,'Value')==1;
        doS2P  = get(chkSaveS2P,'Value')==1;

        if ~doPerm && ~doS2P
            errordlg('Enable at least one: Save CSV (Permittivity) or Save S2P.','Nothing selected');
            return;
        end

        try
            generateMergedOutputsGrouped(combos, mp, state.rootDir, doPerm, doS2P);
        catch ME
            errordlg(['Generation error: ' ME.message], 'Generation error');
            return;
        end

        try
            writeSelectionsLog(combos, state.rootDir, doPerm, doS2P);
        catch ME
            errordlg(['Log write error: ' ME.message], 'Log error');
            return;
        end

        selections = combos;
        uiresume(fig);
        if isvalid(fig), close(fig); end
    end

    % ===================== CORE HELPERS =====================

    function [fileIds, permFileNames, s2pFileNames, cutHz, methodPathAbs, method] = getCurrentSelection()
        fileIds = [];
        permFileNames = {};
        s2pFileNames  = {};
        cutHz = [];
        methodPathAbs = '';
        method = struct();

        mIdx = get(popMethod,'Value');
        sIdx = get(popSample,'Value');
        method = state.methods(mIdx);

        samplePath = fullfile(state.rootDir, state.sampleNames{sIdx});
        methodPathAbs = fullfile(samplePath, char(method.FolderName));
        if ~isfolder(methodPathAbs)
            methodPathAbs = '';
            return;
        end

        cutHz = zeros(numel(state.selBandsSorted),2);
        for i=1:numel(state.selBandsSorted)
            cutHz(i,:) = resolveBandCutoffHz_jsonOnly(method, state.selBandsSorted{i});
        end

        nB = numel(state.bandSelectors);
        fileIds = zeros(1,nB);
        permFileNames = cell(nB,1);
        s2pFileNames  = cell(nB,1);

        for iB=1:nB
            row = state.bandSelectors(iB);
            if isempty(row.ids)
                fileIds(iB) = NaN;
                permFileNames{iB} = '';
                s2pFileNames{iB}  = '';
                continue;
            end
            v = get(row.popup,'Value');
            v = max(1, min(v, numel(row.ids)));
            fileIds(iB) = row.ids(v);

            band = row.band;
            permFileNames{iB} = resolveFileNameById(band, fileIds(iB), 'perm');
            s2pFileNames{iB}  = resolveFileNameById(band, fileIds(iB), 's2p');
        end
    end

    function name = resolveFileNameById(band, id, kind)
        name = '';
        mIdx = get(popMethod,'Value');
        sIdx = get(popSample,'Value');
        method = state.methods(mIdx);

        samplePath = fullfile(state.rootDir, state.sampleNames{sIdx});
        methodPathAbs = fullfile(samplePath, char(method.FolderName));
        if ~isfolder(methodPathAbs), return; end

        bandPath = fullfile(methodPathAbs, band);
        if ~isfolder(bandPath), return; end

        if strcmpi(kind,'perm')
            if ~isfield(method,'PermittivityInput'), return; end
            exts = method.PermittivityInput.Extensions;
            files = listFilesByExtSmart(bandPath, exts);
        else
            if ~isfield(method,'SParameterInput'), return; end
            exts = method.SParameterInput.Extensions;
            files = listFilesByExtSmart(bandPath, exts);
        end

        if isempty(files), return; end
        for k=1:numel(files)
            fid = getLastNumberId(files(k).name);
            if ~isnan(fid) && fid == id
                name = files(k).name;
                return;
            end
        end
    end

    function [ids, names] = listBandFilesAsIds(band)
        ids = [];
        names = {};

        mIdx = get(popMethod,'Value');
        sIdx = get(popSample,'Value');
        method = state.methods(mIdx);

        samplePath = fullfile(state.rootDir, state.sampleNames{sIdx});
        methodPathAbs = fullfile(samplePath, char(method.FolderName));
        bandPath = fullfile(methodPathAbs, band);
        if ~isfolder(bandPath), return; end

        files = [];
        if isfield(method,'PermittivityInput')
            files = listFilesByExtSmart(bandPath, method.PermittivityInput.Extensions);
        end
        if isempty(files) && isfield(method,'SParameterInput')
            files = listFilesByExtSmart(bandPath, method.SParameterInput.Extensions);
        end

        if isempty(files), return; end
        [ids, names] = extractIdNamePairs(files);
        if isempty(ids), return; end

        [idsU, ia] = unique(ids,'stable');
        namesU = names(ia);

        [ids, ord] = sort(idsU(:)');
        names = namesU(ord);
    end

    function used = getUsedIdsForBand(band)
        b = char(band);
        if state.usedIdsMap.isKey(b)
            used = state.usedIdsMap(b);
        else
            used = [];
        end
    end

    function markUsedId(band, id)
        b = char(band);
        used = getUsedIdsForBand(b);
        used = unique([used(:); id]);
        state.usedIdsMap(b) = used(:);
    end

    function strs = makeIdDisplayStrings(ids, names)
        strs = cell(numel(ids),1);
        for i=1:numel(ids)
            strs{i} = sprintf('%d  |  %s', ids(i), names{i});
        end
    end

    function [valIdx, defaultId] = defaultPopupIndex(idsAvail)
        if any(idsAvail==1)
            defaultId = 1;
        else
            defaultId = min(idsAvail);
        end
        valIdx = find(idsAvail==defaultId, 1, 'first');
        if isempty(valIdx), valIdx = 1; end
    end

    function s = selectionSummaryLine(bands, ids, names)
        parts = cell(numel(bands),1);
        for i=1:numel(bands)
            nm = '';
            if i <= numel(names) && ~isempty(names{i})
                nm = names{i};
            end
            parts{i} = sprintf('%s:%d', bands{i}, ids(i));
            if ~isempty(nm)
                parts{i} = sprintf('%s (%s)', parts{i}, nm);
            end
        end
        s = strjoin(parts, ' | ');
    end

    % ===================== GENERATE + LOG (GROUPED) =====================

    function generateMergedOutputsGrouped(combosIn, mpCfg, rootDirAbs, doPerm, doS2P)
        out = mpCfg.Output;

        permMergedSuffix = getFieldOrDefault(out, 'PermittivityMergedSuffix', '_Merged_Data.csv');
        permAvgSuffix    = getFieldOrDefault(out, 'PermittivityAveragedSuffix', '_Averaged_Data.csv');
        s2pMergedSuffix  = getFieldOrDefault(out, 'S2PMergedSuffix', '_S2PMerged.csv');
        s2pAvgSuffix     = getFieldOrDefault(out, 'S2PAveragedSuffix', '_S2PAveraged_Data.csv');

        if isfield(out,"PermittivityMergedHeaders"), permMergedHeaders = out.PermittivityMergedHeaders;
        else, permMergedHeaders = {'Freq_Hz','EpsReal','TanDelta','Thickness_mm'}; end

        if isfield(out,"PermittivityAveragedHeaders"), permAvgHeaders = out.PermittivityAveragedHeaders;
        else, permAvgHeaders = {'Freq_Hz','EpsReal_Mean','EpsReal_STD','TanDelta_Mean','TanDelta_STD','Thickness_mm'}; end

        if isfield(out,"S2PHeaders"), s2pHeaders = out.S2PHeaders;
        else, s2pHeaders = {'Freq_Hz','S11_mag','S11_phase','S21_mag','S21_phase','S12_mag','S12_phase','S22_mag','S22_phase'}; end

        % S2P averaged headers (if not in JSON -> use same as S2PHeaders)
        if isfield(out,"S2PAveragedHeaders"), s2pAvgHeaders = out.S2PAveragedHeaders;
        else, s2pAvgHeaders = s2pHeaders; end

        % ---- group keys: SampleRel + BandsCombo (order-sensitive already sorted) ----
        n = numel(combosIn);
        keys = cell(n,1);
        for i=1:n
            c = combosIn(i);
            keys{i} = [c.SampleRel '||' strjoin(c.BandNames,'_')];
        end
        [uKeys, ~, g] = unique(keys,'stable');

        for gi = 1:numel(uKeys)
            idxs = find(g==gi);
            group = combosIn(idxs);

            c0 = group(1);
            methodPathAbs = fullfile(rootDirAbs, c0.SampleRel); % <root>/<sample>/<methodFolder>
            if ~isfolder(methodPathAbs)
                warning('Method path missing: %s', methodPathAbs);
                continue;
            end

            method = findMethodConfig(mpCfg, c0);
            if isempty(method), continue; end

            comboName = strjoin(c0.BandNames, '_');
            outBandFolder = fullfile(methodPathAbs, 'MergedBands', comboName);
            if ~isfolder(outBandFolder), mkdir(outBandFolder); end

            relPath = strrep(outBandFolder, [rootDirAbs filesep], '');
            outputBaseName = strjoin(strsplit(relPath, filesep), '_');

            % ================= PERMITTIVITY: MERGED + AVERAGED =================
            if doPerm && isfield(method,'PermittivityInput')
                permCfg = method.PermittivityInput;

                stitchedList = cell(numel(group),1);
                for k=1:numel(group)
                    permNames = group(k).PermFileNames;
                    if isempty(permNames) || any(cellfun(@isempty, permNames))
                        stitchedList{k} = [];
                        continue;
                    end
                    stitchedList{k} = stitchPermFromFileNames(methodPathAbs, c0.BandNames, c0.CutoffsHz, permCfg, permNames);
                end
                stitchedList = stitchedList(~cellfun(@isempty, stitchedList));

                if ~isempty(stitchedList)
                    % MERGED: side-by-side (4-col per set)
                    mergedMat = [];
                    for k=1:numel(stitchedList)
                        mergedMat = padAndHcat(mergedMat, stitchedList{k});
                    end

                    colLabels = strings(1, numel(stitchedList) * numel(permMergedHeaders));
                    cc = 1;
                    for k=1:numel(stitchedList)
                        for h=1:numel(permMergedHeaders)
                            colLabels(cc) = sprintf('%d_%s', k, permMergedHeaders{h});
                            cc = cc + 1;
                        end
                    end
                    writetable(array2table(mergedMat, 'VariableNames', cellstr(colLabels)), ...
                        fullfile(outBandFolder, [outputBaseName permMergedSuffix]));

                    % AVERAGED: mean/std across sets per frequency
                    allData = vertcat(stitchedList{:}); % [f eps tan thickMarker]
                    [uf,~,grpF] = unique(allData(:,1));

                    meanEps = NaN(numel(uf),1); stdEps = NaN(numel(uf),1);
                    meanTan = NaN(numel(uf),1); stdTan = NaN(numel(uf),1);
                    thkOut  = NaN(numel(uf),1);

                    for ii=1:numel(uf)
                        rows = allData(grpF==ii, :);
                        epsv = rows(:,2);
                        tanv = rows(:,3);
                        thkv = rows(:,4);

                        epsv(epsv<0) = NaN;        % ignore negative eps
                        tanv(~isfinite(tanv)) = NaN;

                        meanEps(ii) = mean(epsv,'omitnan');
                        stdEps(ii)  = std(epsv,0,'omitnan');
                        meanTan(ii) = mean(tanv,'omitnan');
                        stdTan(ii)  = std(tanv,0,'omitnan');

                        t = thkv(~isnan(thkv));
                        if ~isempty(t)
                            thkOut(ii) = mean(t,'omitnan'); % marker-style thickness
                        end
                    end

                    averagedData = [uf, meanEps, stdEps, meanTan, stdTan, thkOut];
                    writetable(array2table(averagedData, 'VariableNames', permAvgHeaders), ...
                        fullfile(outBandFolder, [outputBaseName permAvgSuffix]));
                end
            end

            % ================= S2P: MERGED + AVERAGED =================
            if doS2P && isfield(method,'SParameterInput')
                spCfg = method.SParameterInput;

                stitchedS = cell(numel(group),1);
                for k=1:numel(group)
                    s2pNames = group(k).S2PFileNames;
                    if isempty(s2pNames) || any(cellfun(@isempty, s2pNames))
                        stitchedS{k} = [];
                        continue;
                    end
                    stitchedS{k} = stitchS2PFromFileNames(methodPathAbs, c0.BandNames, c0.CutoffsHz, spCfg, s2pNames);
                end
                stitchedS = stitchedS(~cellfun(@isempty, stitchedS));

                if ~isempty(stitchedS)
                    % MERGED: side-by-side (9-col per set)
                    mergedS2P = [];
                    for k=1:numel(stitchedS)
                        mergedS2P = padAndHcat(mergedS2P, stitchedS{k});
                    end

                    colLabelsS2P = strings(1, numel(stitchedS) * numel(s2pHeaders));
                    cc = 1;
                    for k=1:numel(stitchedS)
                        for h=1:numel(s2pHeaders)
                            colLabelsS2P(cc) = sprintf('%d_%s', k, s2pHeaders{h});
                            cc = cc + 1;
                        end
                    end
                    writetable(array2table(mergedS2P, 'VariableNames', cellstr(colLabelsS2P)), ...
                        fullfile(outBandFolder, [outputBaseName s2pMergedSuffix]));

                    % AVERAGED: complex-mean per frequency per Sij
                    allS = vertcat(stitchedS{:}); % [f, S11mag,S11ph, S21mag,S21ph, S12mag,S12ph, S22mag,S22ph]
                    [uf,~,grpF] = unique(allS(:,1));
                    avgS = NaN(numel(uf), 9);
                    avgS(:,1) = uf;

                    for ii=1:numel(uf)
                        rows = allS(grpF==ii, :);

                        % helper to complex-average (mag,phase_deg)
                        [m, p] = complexMeanMagPhase(rows(:,2), rows(:,3)); avgS(ii,2)=m; avgS(ii,3)=p;
                        [m, p] = complexMeanMagPhase(rows(:,4), rows(:,5)); avgS(ii,4)=m; avgS(ii,5)=p;
                        [m, p] = complexMeanMagPhase(rows(:,6), rows(:,7)); avgS(ii,6)=m; avgS(ii,7)=p;
                        [m, p] = complexMeanMagPhase(rows(:,8), rows(:,9)); avgS(ii,8)=m; avgS(ii,9)=p;
                    end

                    writetable(array2table(avgS, 'VariableNames', s2pAvgHeaders), ...
                        fullfile(outBandFolder, [outputBaseName s2pAvgSuffix]));
                end
            end

            fprintf('✅ Generated under: %s\n', outBandFolder);
        end
    end

    function [magOut, phOutDeg] = complexMeanMagPhase(magVec, phDegVec)
        magVec(~isfinite(magVec)) = NaN;
        phDegVec(~isfinite(phDegVec)) = NaN;

        % build complex
        z = magVec .* exp(1j*deg2rad(phDegVec));
        z(~isfinite(z)) = NaN;

        zMean = mean(z,'omitnan');
        if ~isfinite(zMean)
            magOut = NaN; phOutDeg = NaN; return;
        end
        magOut = abs(zMean);
        phOutDeg = rad2deg(angle(zMean));
    end

    function writeSelectionsLog(combosIn, rootDirAbs, doPerm, doS2P)
        ts = datestr(now,'yyyymmdd_HHMMSS');
        logPath = fullfile(rootDirAbs, ['MergedBands_Selections_' ts '.txt']);
        fid = fopen(logPath,'w');
        if fid==-1
            error('Could not write log: %s', logPath);
        end

        fprintf(fid,'Band Stitch Selections Log\n');
        fprintf(fid,'Timestamp: %s\n', datestr(now));
        fprintf(fid,'RootDir: %s\n', rootDirAbs);
        fprintf(fid,'Save CSV (Permittivity): %d\n', doPerm);
        fprintf(fid,'Save S2P:               %d\n\n', doS2P);

        for i=1:numel(combosIn)
            c = combosIn(i);
            fprintf(fid,'[%d]\n', i);
            fprintf(fid,'  Sample/Method: %s\n', c.SampleRel);
            fprintf(fid,'  Bands: %s\n', strjoin(c.BandNames, ', '));
            fprintf(fid,'  FileIds: %s\n', mat2str(c.FileIds));

            if doPerm
                fprintf(fid,'  Perm files:\n');
                for b=1:numel(c.BandNames)
                    fprintf(fid,'    - %s: %s\n', c.BandNames{b}, c.PermFileNames{b});
                end
            end
            if doS2P
                fprintf(fid,'  S2P files:\n');
                for b=1:numel(c.BandNames)
                    fprintf(fid,'    - %s: %s\n', c.BandNames{b}, c.S2PFileNames{b});
                end
            end

            fprintf(fid,'  CutoffsHz:\n');
            for b=1:size(c.CutoffsHz,1)
                fprintf(fid,'    - %s: %.12g .. %.12g (Hz)\n', c.BandNames{b}, c.CutoffsHz(b,1), c.CutoffsHz(b,2));
            end
            fprintf(fid,'\n------------------------------------------------------------\n\n');
        end

        fclose(fid);
        fprintf('📝 Log saved: %s\n', logPath);
    end

    % ===================== STITCHING =====================

    function d = stitchPermFromFileNames(methodPathAbs, bandNames, cutHz, permCfg, fileNamesOneSet)
        nBands = numel(bandNames);
        parts = cell(nBands,1);
        thickVal = NaN;

        for j=1:nBands
            bandPath = fullfile(methodPathAbs, bandNames{j});
            fp = smartFileResolve(bandPath, fileNamesOneSet{j});
            if isempty(fp), d=[]; return; end

            raw = parsePermittivityFile_withThickness(fp, permCfg); % [f eps tan thickMarker]
            if isempty(raw), d=[]; return; end

            low = cutHz(j,1); up = cutHz(j,2);
            if j < nBands
                nextLow = cutHz(j+1,1);
                raw = raw(raw(:,1) >= low & raw(:,1) <= up & raw(:,1) < nextLow, :);
            else
                raw = raw(raw(:,1) >= low & raw(:,1) <= up, :);
            end

            if ~isempty(raw) && isnan(thickVal)
                tv = raw(:,4); tv = tv(~isnan(tv));
                if ~isempty(tv), thickVal = tv(1); end
            end

            parts{j} = raw(:,1:3); % drop marker for concatenation
        end

        d = vertcat(parts{:});
        if isempty(d), return; end
        d = sortrows(d,1);

        thCol = NaN(size(d,1),1);
        if ~isnan(thickVal), thCol(1) = thickVal; end
        d = [d(:,1), d(:,2), d(:,3), thCol];
    end

    function d = stitchS2PFromFileNames(methodPathAbs, bandNames, cutHz, spCfg, fileNamesOneSet)
        nBands = numel(bandNames);
        parts = cell(nBands,1);

        for j=1:nBands
            bandPath = fullfile(methodPathAbs, bandNames{j});
            fp = smartFileResolve(bandPath, fileNamesOneSet{j});
            if isempty(fp), d=[]; return; end

            raw = readS2PAsMatrix(fp, spCfg.Columns, spCfg);
            if isempty(raw), d=[]; return; end

            low = cutHz(j,1); up = cutHz(j,2);
            if j < nBands
                nextLow = cutHz(j+1,1);
                raw = raw(raw(:,1) >= low & raw(:,1) <= up & raw(:,1) < nextLow, :);
            else
                raw = raw(raw(:,1) >= low & raw(:,1) <= up, :);
            end

            parts{j} = raw;
        end

        d = vertcat(parts{:});
        if isempty(d), return; end
        d = sortrows(d,1);
    end

    % ===================== FILE DISCOVERY =====================

    function fp = smartFileResolve(bandPath, fileName)
        fp = '';
        if isempty(fileName), return; end

        cand = fullfile(bandPath, fileName);
        if isfile(cand)
            fp = cand; return;
        end

        cand2 = fullfile(bandPath, 'Merged_Filtered_Revision', fileName);
        if isfile(cand2)
            fp = cand2; return;
        end

        dd2 = dir(fullfile(bandPath,'**',fileName));
        if ~isempty(dd2)
            fp = fullfile(dd2(1).folder, dd2(1).name);
        end
    end

    function files = listFilesByExtSmart(bandPath, extensions)
        files = [];
        files1 = listFilesByExt(bandPath, extensions);

        sub = fullfile(bandPath,'Merged_Filtered_Revision');
        files2 = [];
        if isfolder(sub)
            files2 = listFilesByExt(sub, extensions);
        end

        files = [files1; files2];
        if isempty(files), return; end

        fulls = arrayfun(@(d)fullfile(d.folder,d.name), files, 'UniformOutput', false);
        [~, ia] = unique(fulls,'stable');
        files = files(ia);
    end

    % ===================== JSON / UTIL =====================

    function cut = resolveBandCutoffHz_jsonOnly(method, bandName)
        cut = [NaN NaN];
        bands = method.Bands;
        hit = [];
        for k=1:numel(bands)
            if strcmpi(string(bands(k).Name), string(bandName))
                hit = bands(k); break;
            end
        end
        if isempty(hit)
            error('Band not found in JSON: %s', string(bandName));
        end
        if isfield(hit,'LowerCutoffHz') && isfield(hit,'UpperCutoffHz')
            cut(1) = double(hit.LowerCutoffHz);
            cut(2) = double(hit.UpperCutoffHz);
        else
            error("Band '%s' must have LowerCutoffHz/UpperCutoffHz in JSON.", string(bandName));
        end
    end

    function method = findMethodConfig(mpCfg, combo)
        method = [];
        for k=1:numel(mpCfg.Methods)
            if isfield(mpCfg.Methods(k),'FolderName') && strcmp(mpCfg.Methods(k).FolderName, combo.MethodFolderName)
                method = mpCfg.Methods(k); return;
            end
        end
        for k=1:numel(mpCfg.Methods)
            if isfield(mpCfg.Methods(k),'Name') && strcmp(mpCfg.Methods(k).Name, combo.MethodName)
                method = mpCfg.Methods(k); return;
            end
        end
    end

    function [ids, names] = extractIdNamePairs(files)
        ids = [];
        names = {};
        for k=1:numel(files)
            id = getLastNumberId(files(k).name);
            if ~isnan(id)
                ids(end+1,1) = id; %#ok<AGROW>
                names{end+1,1} = files(k).name; %#ok<AGROW>
            end
        end
    end

    function id = getLastNumberId(filename)
        [~, n, ~] = fileparts(filename);
        tok = regexp(n, '(\d+)(?!.*\d)', 'tokens', 'once');
        if isempty(tok)
            id = NaN;
        else
            id = str2double(tok{1});
        end
    end

    function files = listFilesByExt(folderPath, extensions)
        extensions = lower(string(extensions));
        dd = dir(fullfile(folderPath,'*'));
        dd = dd(~[dd.isdir]);
        if isempty(dd), files = dd; return; end
        [~,~,e] = cellfun(@fileparts, {dd.name}, 'UniformOutput', false);
        e = lower(string(e));
        files = dd(ismember(e, extensions));
        for k=1:numel(files)
            files(k).folder = folderPath;
        end
    end

    function val = getFieldOrDefault(S, name, defaultVal)
        if isfield(S, name) && ~isempty(S.(name))
            v = S.(name);
            if isstring(v), v = char(v); end
            val = v;
        else
            val = defaultVal;
        end
    end

    function merged = padAndHcat(existing, A)
        if isempty(existing)
            merged = A;
            return;
        end
        nRows = size(A,1);
        eRows = size(existing,1);
        maxRows = max(nRows, eRows);
        if eRows < maxRows
            existing(end+1:maxRows,:) = NaN;
        end
        if nRows < maxRows
            A(end+1:maxRows,:) = NaN;
        end
        merged = [existing A];
    end

    % ===================== PERM READER (WITH THICKNESS MARKER) =====================

    function numericData = parsePermittivityFile_withThickness(filePath, permCfg)
        % output: [Freq_Hz, EpsReal, TanDelta, Thickness_mm_marker]
        fileContents = readcell(filePath);

        skip = 0;
        if isfield(permCfg,'HeaderRowsToSkip'), skip = permCfg.HeaderRowsToSkip; end
        dataBlock = fileContents(skip+1:end, :);

        cols = [permCfg.FrequencyColumn, permCfg.EpsRealColumn, permCfg.TanColumn];
        raw = dataBlock(:, cols);

        numeric = nan(size(raw,1),3);
        for r=1:size(raw,1)
            for c=1:3
                v = raw{r,c};
                if isnumeric(v), numeric(r,c)=double(v);
                else, numeric(r,c)=str2double(string(v));
                end
            end
        end

        numeric = rmmissing(numeric);
        if isempty(numeric), numericData=[]; return; end

        freq    = numeric(:,1);
        epsReal = numeric(:,2);
        col3    = numeric(:,3);

        % TanDelta vs EpsImag
        if isfield(permCfg,'TanDelta_EpsImag')
            mode = lower(string(permCfg.TanDelta_EpsImag));
            if mode == "epsimag"
                tanDelta = col3 ./ epsReal;
                tanDelta(~isfinite(tanDelta)) = NaN;
            else
                tanDelta = col3;
            end
        else
            tanDelta = col3;
        end

        % freq unit conversion
        if isfield(permCfg,'FreqUnit')
            u = upper(string(permCfg.FreqUnit));
            switch u
                case "HZ"
                case "KHZ", freq = freq*1e3;
                case "MHZ", freq = freq*1e6;
                case "GHZ", freq = freq*1e9;
                otherwise, error("Unsupported perm FreqUnit: %s", u);
            end
        end

        % thickness marker (only first row)
        thicknessVal = NaN;
        if isfield(permCfg,"Thickness")
            th = permCfg.Thickness;
            if isfield(th,"Mode")
                switch lower(string(th.Mode))
                    case "cell"
                        rawCell = fileContents{th.Row, th.Col};
                        thickStr = string(rawCell);
                        if isfield(th,"RemoveString") && ~isempty(th.RemoveString)
                            thickStr = strrep(thickStr, string(th.RemoveString), "");
                        end
                        thicknessVal = str2double(thickStr);
                    case "constant"
                        if isfield(th,"Value")
                            thicknessVal = double(th.Value);
                        end
                end
            end
        end

        thicknessMarker = NaN(size(freq));
        if ~isnan(thicknessVal)
            thicknessMarker(1) = thicknessVal;
        end

        numericData = [freq, epsReal, tanDelta, thicknessMarker];
    end

    % ===================== S2P READER =====================

    function numericData = readS2PAsMatrix(filePath, columnsCfg, spCfg)
        % output:
        % [Freq_Hz, S11_mag, S11_phase_deg, S21_mag, S21_phase_deg, S12_mag, S12_phase_deg, S22_mag, S22_phase_deg]
        fid = fopen(filePath,'r');
        if fid==-1, error('Cannot open: %s', filePath); end

        lines = {};
        headerFreqUnit = '';
        headerFormat = '';

        defaultFormat = "MA";
        if isfield(spCfg,'DataFormat') && ~isempty(spCfg.DataFormat)
            defaultFormat = upper(string(spCfg.DataFormat));
        end

        defaultFreqUnit = "HZ";
        if isfield(spCfg,'FreqUnit') && ~isempty(spCfg.FreqUnit)
            defaultFreqUnit = upper(string(spCfg.FreqUnit));
        end

        while ~feof(fid)
            line = strtrim(fgetl(fid));
            if ~ischar(line), break; end
            if isempty(line), continue; end

            if startsWith(line,'#')
                tokens = upper(string(strsplit(line)));
                freqTokens = ["HZ","KHZ","MHZ","GHZ"];
                fmtTokens  = ["DB","MA","RI"];
                hitF = freqTokens(ismember(freqTokens,tokens));
                hitM = fmtTokens(ismember(fmtTokens,tokens));
                if ~isempty(hitF), headerFreqUnit = char(hitF(1)); end
                if ~isempty(hitM), headerFormat   = char(hitM(1)); end
                continue;
            end
            if startsWith(line,'!'), continue; end

            lines{end+1} = line; %#ok<AGROW>
        end
        fclose(fid);

        if isempty(lines), numericData=[]; return; end

        data = cellfun(@(l) sscanf(l,'%f')', lines, 'UniformOutput', false);
        data = vertcat(data{:});
        data = rmmissing(data);

        effFreqUnit = defaultFreqUnit;
        if ~isempty(headerFreqUnit), effFreqUnit = upper(string(headerFreqUnit)); end

        freq = data(:, columnsCfg.Frequency);
        switch effFreqUnit
            case "HZ"
            case "KHZ", freq = freq*1e3;
            case "MHZ", freq = freq*1e6;
            case "GHZ", freq = freq*1e9;
            otherwise, error("Unsupported S2P freq unit: %s", effFreqUnit);
        end

        effFormat = defaultFormat;
        if ~isempty(headerFormat), effFormat = upper(string(headerFormat)); end

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
                error("Unsupported S2P format: %s", effFormat);
        end

        numericData = [freq, S11_mag, S11_ph, S21_mag, S21_ph, S12_mag, S12_ph, S22_mag, S22_ph];
    end
end
