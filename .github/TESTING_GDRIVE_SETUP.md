# Testing Google Drive Setup

## Quick Checklist

Before pushing and testing in CI, verify:

### 1. File Sharing ✓
- [ ] Open your Google Drive file in browser
- [ ] Click "Share" button
- [ ] Verify the service account email (`*@*.iam.gserviceaccount.com`) appears in "Who has access"
- [ ] Permission is set to "Viewer"

### 2. GitHub Secrets ✓
Go to: `https://github.com/YOUR_ORG/iq-to-audio/settings/secrets/actions`

Required secrets:
- [ ] `GDRIVE_SERVICE_ACCOUNT_JSON` - Full JSON content (including `{` and `}`)
- [ ] `GDRIVE_FILE_ID` - File ID from Drive URL (alphanumeric string)
- [ ] `GDRIVE_FILE_SHA256` - SHA256 checksum of your tar.xz file

Optional (but recommended):
- [ ] `GDRIVE_CLIENT_ID` - Your custom OAuth client ID
- [ ] `GDRIVE_CLIENT_SECRET` - Your custom OAuth client secret

### 3. Get File ID from URL

Your Drive URL looks like:
```
https://drive.google.com/file/d/1AbC2DeF3GhI4JkL5MnO6PqR7StU8VwX9/view
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  This is your GDRIVE_FILE_ID
```

### 4. Calculate SHA256

```bash
# macOS
shasum -a 256 testfiles/iq-to-audio-fixtures.tar.xz

# Linux
sha256sum testfiles/iq-to-audio-fixtures.tar.xz
```

Copy the first field (the hash) into `GDRIVE_FILE_SHA256`.

## Commit and Push

Once the secrets are configured:

```bash
git add .github/actions/download-test-fixtures/action.yml
git add scripts/download_test_fixtures.py
git add docs/
git add .github/TESTING_GDRIVE_SETUP.md
git commit -m "Fix rclone for shared files and Windows temp path"
git push
```

## What to Look For in CI Logs

### Success Looks Like:

```
rclone configuration created
Service account file: /tmp/gdrive-service-account.json
Downloading test fixtures from Google Drive...
File ID: ***
Searching for file in shared drive space...
Attempting to download file by ID...
Transferred:   397.000 MiB / 397.000 MiB, 100%, 12.345 MiB/s, ETA 0s
Download complete: testfiles/iq-to-audio-fixtures.tar.xz
-rw-r--r-- 1 runner docker 397M Oct 13 13:45 testfiles/iq-to-audio-fixtures.tar.xz
Verifying SHA256 checksum...
Expected: abc123...
Actual:   abc123...
Checksum verified successfully
```

### Common Errors:

**"directory not found"** = File not shared with service account
- Solution: Share the file in Google Drive

**"The system cannot find the path specified"** = Windows temp path issue (should be fixed now)

**"Checksum mismatch"** = Wrong file or corrupted upload
- Solution: Re-upload file to Drive and update SHA256 secret

**"Rate limited"** = Using rclone's public client too much
- Solution: Add `GDRIVE_CLIENT_ID` and `GDRIVE_CLIENT_SECRET` secrets

## Test Locally (Optional)

```bash
# Export secrets as environment variables
export GDRIVE_SERVICE_ACCOUNT_JSON='<paste JSON here>'
export GDRIVE_FILE_ID='your-file-id'
export GDRIVE_FILE_SHA256='your-checksum'
export GDRIVE_CLIENT_ID='your-client-id'  # Optional
export GDRIVE_CLIENT_SECRET='your-secret'  # Optional

# Install rclone
brew install rclone  # macOS
# or
sudo apt-get install rclone  # Linux

# Run download script
python scripts/download_test_fixtures.py

# Or let pytest download automatically
uv run pytest -v
```

## Verifying the Service Account Email

1. Open your service account JSON file
2. Look for the `client_email` field:
   ```json
   {
     "type": "service_account",
     "project_id": "your-project",
     "client_email": "iq-to-audio-ci@your-project.iam.gserviceaccount.com",
     ...
   }
   ```
3. Copy that exact email
4. In Google Drive, verify it matches what you shared

## If Still Failing

Add `-vvv` to rclone commands in the action to see verbose output:

```bash
rclone copyto \
  --drive-shared-with-me \
  --drive-acknowledge-abuse \
  --progress \
  --transfers 1 \
  "gdrive:{file_id}" \
  "destination" \
  -vvv  # Maximum verbosity
```

This will show exactly what rclone sees in the Drive account.
