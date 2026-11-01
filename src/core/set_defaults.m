function cfg = set_defaults(cfg, def)
% SET_DEFAULTS  Fill the fields of cfg that are missing with those of def, and
% reject fields that def does not know (catches misspelled options).
if isempty(cfg), cfg = struct(); end
fc = fieldnames(cfg);
if ~isempty(fc)                      % isfield(def, {}) is a scalar false
    bad = fc(~isfield(def, fc));
    if ~isempty(bad), error('unknown option(s): %s', strjoin(bad', ', ')); end
end
fd = fieldnames(def);
for i = 1:numel(fd)
    if ~isfield(cfg, fd{i}), cfg.(fd{i}) = def.(fd{i}); end
end
end
