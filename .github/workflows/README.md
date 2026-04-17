# GitHub Actions Workflows

## Release Workflow

The `release.yml` workflow automatically builds and uploads UCCL wheels to PyPI when a release is created.

### Triggers

- **Automatic**: Triggers when a new GitHub release is created
- **Manual**: Can be triggered manually via GitHub Actions UI with custom parameters

### Setup

1. **Create PyPI API Token**:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token (scope: entire account or specific project)
   - Copy the token

2. **Add Secret to GitHub**:
   - Go to your repository Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token
   - Click "Add secret"

### Usage

#### Automatic Release

1. Create a new release on GitHub:
   - Go to Releases → Draft a new release
   - Tag version (e.g., `v0.0.1`)
   - Release title (e.g., `v0.0.1`)
   - Publish release

2. The workflow will automatically:
   - Build wheels for all targets (cuda, rocm) and Python versions (3.8-3.12)
   - Validate all wheels
   - Upload to PyPI
   - Create artifacts for download

#### Manual Release

1. Go to Actions → Release to PyPI
2. Click "Run workflow"
3. Configure options:
   - **Python versions**: Space-separated list (default: `3.8 3.9 3.10 3.11 3.12`)
   - **Targets**: Space-separated list (default: `cuda rocm`)
   - **PyPI repository**: `pypi` or `testpypi` (default: `pypi`)
   - **Dry run**: Check to build wheels without uploading (recommended for testing)
4. Click "Run workflow"

#### Testing the Workflow

**⚠️ Important:** Always test before your first production release!

**Recommended testing approach:**
1. **First test**: Use **Dry run** mode (builds wheels but doesn't upload)
2. **Second test**: Use **TestPyPI** (safe testing environment)
3. **Production**: Create a GitHub release (uploads to production PyPI)

See [TESTING.md](./TESTING.md) for detailed testing instructions.

### Workflow Steps

1. **Checkout**: Checks out the repository code
2. **Setup Python**: Sets up Python 3.11 for running scripts
3. **Install dependencies**: Installs twine and build tools
4. **Setup Docker**: Configures Docker Buildx for building wheels
5. **Build wheels**: Builds wheels for all specified targets and Python versions
6. **Validate wheels**: Validates all wheels using twine check
7. **Upload to PyPI**: Uploads validated wheels to PyPI
8. **Upload artifacts**: Saves wheels as GitHub artifacts for download

### Output

- Wheels are uploaded to PyPI and can be installed with `pip install uccl`
- All wheels are also saved as GitHub artifacts for 30 days
- A summary is generated showing all built wheels

### Troubleshooting

#### Build Failures

- Check Docker is properly configured in the workflow
- Verify Docker images can be built (check Dockerfile dependencies)
- Review build logs for specific errors

#### Upload Failures

- Verify `PYPI_API_TOKEN` secret is set correctly
- Check if the version already exists on PyPI (can't re-upload same version)
- Ensure wheels pass validation (check twine check output)

#### Missing Wheels

- Check that all targets and Python versions are supported
- Verify build completed successfully for all combinations
- Check for architecture-specific issues

### Security

- PyPI tokens are stored as GitHub secrets (encrypted)
- Tokens are only accessible during workflow execution
- Use project-scoped tokens when possible for better security

### Example

To release version 0.0.1:

1. Update version in `uccl/__init__.py`:
   ```python
   __version__ = "0.0.1"
   ```

2. Commit and push:
   ```bash
   git add uccl/__init__.py
   git commit -m "Release v0.0.1"
   git push
   ```

3. Create GitHub release:
   - Tag: `v0.0.1`
   - Title: `v0.0.1`
   - Publish release

4. Workflow runs automatically and uploads to PyPI

5. Verify installation:
   ```bash
   pip install uccl==0.0.1
   ```

