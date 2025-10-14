# Custom PyInstaller hook for webrtcvad
#
# This hook overrides the default hook from _pyinstaller_hooks_contrib because
# we use webrtcvad-wheels (not webrtcvad), which has different package metadata.
#
# The default hook tries to call copy_metadata('webrtcvad') which fails because
# the installed package is actually 'webrtcvad-wheels'.

from PyInstaller.utils.hooks import copy_metadata

# Try to copy metadata from webrtcvad-wheels first, fall back to webrtcvad
try:
    datas = copy_metadata('webrtcvad-wheels')
except Exception:
    # If webrtcvad-wheels metadata isn't found, try webrtcvad
    try:
        datas = copy_metadata('webrtcvad')
    except Exception:
        # If neither works, just skip metadata collection
        # The module will still be imported correctly
        datas = []

# Ensure the _webrtcvad native module is included
hiddenimports = ['_webrtcvad']
