# Marks TSDiNO/utils as a regular package.
#
# Without this file it is only a *namespace* package, which Python ranks below a
# real module of the same name found later on sys.path. When run_softclt puts
# softclt_ts2vec (which contains utils.py) on the path, `from utils.util import ...`
# in TSDiNO/models/patchTST.py would resolve to SoftCLT's utils.py and fail with
# "No module named 'utils.util'; 'utils' is not a package".
