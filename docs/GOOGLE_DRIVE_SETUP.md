# Google Drive Setup for Test Fixtures

This guide explains how to configure Google Drive with rclone to download test fixtures in GitHub Actions and locally, replacing the Git LFS approach.

## Overview

Test fixtures (`iq-to-audio-fixtures.tar.xz`, 397MB) are stored in Google Drive instead of Git LFS to:
- Avoid GitHub Actions LFS bandwidth limits
- Reduce CI build times
- Enable faster downloads via Google Drive's CDN

The setup uses **rclone** with **service account authentication** for non-interactive downloads in CI/CD.

## Prerequisites

- Google account with Google Drive access
- Google Cloud Console access (free tier is sufficient)
- Admin access to your GitHub repository (to set secrets)

## Part 1: Upload Test Fixtures to Google Drive

1. **Upload the file**:
   - Upload `testfiles/iq-to-audio-fixtures.tar.xz` to your Google Drive
   - Note: You can place it in any folder or at the root level

2. **Get the file ID**:
   - Right-click the file in Google Drive
   - Select "Get link" → "Anyone with the link can view"
   - The file ID is the part between `/d/` and `/view` in the URL:
     ```
     https://drive.google.com/file/d/1AbC2DeF3GhI4JkL5MnO6PqR7StU8VwX9/view
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                       This is your GDRIVE_FILE_ID
     ```
   - Save this ID for later

3. **Calculate SHA256 checksum**:
   ```bash
   # macOS
   shasum -a 256 testfiles/iq-to-audio-fixtures.tar.xz

   # Linux
   sha256sum testfiles/iq-to-audio-fixtures.tar.xz
   ```
   - Save this checksum for later (you'll need it for `GDRIVE_FILE_SHA256`)

## Part 2: Create Google Cloud Project and Service Account

### 2.1 Create/Select a Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note the project name/ID

### 2.2 Enable Google Drive API

1. In Google Cloud Console, go to **APIs & Services** → **Library**
2. Search for "Google Drive API"
3. Click **Enable**

### 2.3 Create Service Account

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **Service Account**
3. Fill in details:
   - **Service account name**: `iq-to-audio-ci` (or any name you prefer)
   - **Service account ID**: Will auto-generate
   - **Description**: "Service account for downloading test fixtures in CI"
4. Click **Create and Continue**
5. Skip the optional steps (no roles needed for Drive access)
6. Click **Done**

### 2.4 Create Service Account Key

1. Find your new service account in the list
2. Click on the service account email
3. Go to **Keys** tab
4. Click **Add Key** → **Create new key**
5. Choose **JSON** format
6. Click **Create**
7. A JSON file will download automatically
8. **Keep this file secure!** It contains credentials for accessing your Drive

### 2.5 Share Drive File with Service Account

**⚠️ CRITICAL STEP**: The service account must have access to the file!

1. Open the JSON key file you just downloaded
2. Find the `client_email` field (looks like `iq-to-audio-ci@project-id.iam.gserviceaccount.com`)
3. **Copy the entire email address**
4. In Google Drive, **right-click** your `iq-to-audio-fixtures.tar.xz` file
5. Select **Share**
6. **Paste the service account email address** into the "Add people and groups" field
7. Set permission to **Viewer** (default)
8. Uncheck "Notify people" (optional, to avoid sending an email to the service account)
9. Click **Share**

**Verification**:
- The file should now show the service account email in the "Who has access" section
- The permission should be "Viewer"
- If the file is in a folder, you don't need to share the folder—just the file is enough

**Common issues**:
- If you don't share the file, you'll get "directory not found" errors
- Make sure you're sharing the actual file, not a shortcut to it
- The service account email must be exactly as it appears in the JSON file

## Part 3: (Optional) Create Custom OAuth Client to Avoid Rate Limits

**Note**: This step is optional but **highly recommended** to avoid rclone's public client rate limits.

### 3.1 Create OAuth Consent Screen

1. In Google Cloud Console, go to **APIs & Services** → **OAuth consent screen**
2. Choose **External** user type
3. Fill in required fields:
   - **App name**: `iq-to-audio` (or any name)
   - **User support email**: Your email
   - **Developer contact email**: Your email
4. Click **Save and Continue**
5. On "Scopes" page, click **Save and Continue** (no scopes needed)
6. On "Test users" page, click **Save and Continue**
7. Review and click **Back to Dashboard**

### 3.2 Create OAuth Client ID

1. Go to **APIs & Services** → **Credentials**
2. Click **Create Credentials** → **OAuth client ID**
3. Choose **Desktop app** as application type
4. Name it `iq-to-audio-client` (or any name)
5. Click **Create**
6. **Important**: Copy the **Client ID** and **Client Secret**
7. Save these for the next step

## Part 4: Configure GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each of the following:

### Required Secrets

| Secret Name | Value | How to Get It |
|-------------|-------|---------------|
| `GDRIVE_SERVICE_ACCOUNT_JSON` | Full contents of the service account JSON file | Paste the entire JSON file contents (from Part 2.4) |
| `GDRIVE_FILE_ID` | Google Drive file ID | From Part 1, step 2 |
| `GDRIVE_FILE_SHA256` | SHA256 checksum | From Part 1, step 3 |

### Optional Secrets (Recommended)

| Secret Name | Value | How to Get It |
|-------------|-------|---------------|
| `GDRIVE_CLIENT_ID` | OAuth client ID | From Part 3.2, step 6 |
| `GDRIVE_CLIENT_SECRET` | OAuth client secret | From Part 3.2, step 6 |

**Note**: If you skip the optional secrets, rclone will use its default public client, which may be rate limited under heavy usage.

## Part 5: Verify Setup

### 5.1 Test in GitHub Actions

1. Push a commit or manually trigger the CI workflow
2. Check the workflow logs for the "Download test fixtures from Google Drive" step
3. Look for messages like:
   ```
   Installing rclone...
   rclone configuration created
   Downloading test fixtures from Google Drive...
   Download complete: testfiles/iq-to-audio-fixtures.tar.xz
   Checksum verified successfully
   ```

### 5.2 Test Locally (Optional)

If you want to test locally:

1. Install rclone:
   ```bash
   # macOS
   brew install rclone

   # Linux
   sudo apt-get install rclone
   ```

2. Export environment variables:
   ```bash
   export GDRIVE_SERVICE_ACCOUNT_JSON='<paste JSON contents here>'
   export GDRIVE_FILE_ID='<your file ID>'
   export GDRIVE_FILE_SHA256='<your checksum>'
   export GDRIVE_CLIENT_ID='<your client ID>'  # Optional
   export GDRIVE_CLIENT_SECRET='<your client secret>'  # Optional
   ```

3. Run the download script:
   ```bash
   python scripts/download_test_fixtures.py
   ```

4. Or let pytest download automatically:
   ```bash
   uv run pytest
   ```

## Troubleshooting

### "directory not found" error

**This is the most common error** and means the service account cannot access the file.

**Solutions**:
1. **Verify file sharing**: Open the file in Google Drive, click "Share", and confirm the service account email appears in "Who has access"
2. **Check the service account email**: Open your JSON key file and verify the `client_email` field matches exactly what you shared
3. **Re-share the file**: Remove the service account from sharing and add it again
4. **Check file location**: If the file is in a Team Drive/Shared Drive, you may need different permissions
5. **Test manually**: Try running the download script locally with the same credentials to see detailed error messages

### "Permission denied" or "File not found"

- Verify the service account email has "Viewer" access to the Drive file (see "directory not found" above)
- Check that `GDRIVE_FILE_ID` is correct (it should be the alphanumeric string from the Drive URL)
- Ensure the file is not in a restricted folder
- Make sure you're not sharing a shortcut—share the actual file

### "Rate limited" errors

- Add the optional `GDRIVE_CLIENT_ID` and `GDRIVE_CLIENT_SECRET` secrets
- These use your own OAuth client instead of rclone's public one

### "Checksum mismatch"

- The file in Google Drive might be different from what you expect
- Recalculate the SHA256 and update `GDRIVE_FILE_SHA256` secret
- Ensure the file wasn't corrupted during upload

### rclone not found

- The GitHub Action automatically installs rclone
- For local usage, install it manually (see Part 5.2)

### Service account JSON is invalid

- Verify you copied the entire JSON file contents
- Check for extra whitespace or line breaks
- Ensure the JSON is valid using a JSON validator

## Security Notes

- **Never commit the service account JSON file to git**
- Service account has read-only access to Drive (via "Viewer" role)
- Service account can only access files explicitly shared with it
- GitHub Secrets are encrypted and only exposed to authorized workflows

## Updating the Fixtures

When you need to update the test fixtures:

1. Generate a new tar.xz file:
   ```bash
   XZ_OPT='-9e -T0' tar -C testfiles -cJf testfiles/iq-to-audio-fixtures.tar.xz \
     fc-132334577Hz-ft-132300000-AM.wav \
     fc-456834049Hz-ft-455837500-ft2-456872500-NFM.wav
   ```

2. Upload the new file to Google Drive (replace the old one)

3. Calculate the new SHA256 checksum

4. Update the `GDRIVE_FILE_SHA256` secret in GitHub

5. If you changed the file location, update `GDRIVE_FILE_ID`

## Cost Considerations

- **Google Cloud**: Free tier includes:
  - Service accounts: Free
  - Drive API calls: 1 billion queries/day (way more than needed)
- **Google Drive Storage**: 15 GB free (our 397MB file is well within limits)
- **GitHub Actions**: No LFS bandwidth costs (the old problem we're solving!)

## Support

For issues with:
- **rclone**: See [rclone documentation](https://rclone.org/drive/)
- **Google Cloud/Drive**: See [Google Cloud support](https://cloud.google.com/support)
- **This setup**: Open an issue in the iq-to-audio repository
