# Testing the Release Workflow

This guide explains how to safely test the release workflow without accidentally uploading to production PyPI.

## Safe Testing Methods

### Method 1: Dry Run (Recommended for First Test)

The safest way to test is using the **dry run** option, which builds wheels but doesn't upload them.

**Steps:**
1. Go to **Actions** → **Release to PyPI** in your GitHub repository
2. Click **Run workflow**
3. Configure:
   - **Dry run**: ✅ Check this box (most important!)
   - **Python versions**: `3.10` (start with one version for faster testing)
   - **Targets**: `cuda` (start with one target)
   - **PyPI repository**: `pypi` (doesn't matter since dry run won't upload)
4. Click **Run workflow**

**What happens:**
- ✅ Builds wheels
- ✅ Validates wheels
- ✅ Creates artifacts
- ❌ **Does NOT upload to PyPI**

**Verify:**
- Check the workflow logs to see if builds succeeded
- Download artifacts to verify wheels were built
- Check the summary for build statistics

### Method 2: TestPyPI (Recommended for Full Test)

TestPyPI is a separate testing environment for PyPI packages. It's safe to upload here.

**Setup:**
1. Create a TestPyPI account at https://test.pypi.org (can use same username as PyPI)
2. Create a TestPyPI API token at https://test.pypi.org/manage/account/token/
3. Add it as a GitHub secret named `TESTPYPI_API_TOKEN` (or reuse `PYPI_API_TOKEN`)

**Steps:**
1. Go to **Actions** → **Release to PyPI**
2. Click **Run workflow**
3. Configure:
   - **PyPI repository**: `testpypi` ⚠️ **Important!**
   - **Python versions**: `3.10` (or all versions)
   - **Targets**: `cuda` (or all targets)
   - **Dry run**: Leave unchecked
4. Click **Run workflow**

**What happens:**
- ✅ Builds wheels
- ✅ Validates wheels
- ✅ Uploads to TestPyPI (safe, separate from production)
- ✅ Creates artifacts

**Verify:**
```bash
# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ uccl
```

### Method 3: Limited Build Test

Test with minimal builds to save time and resources.

**Steps:**
1. Go to **Actions** → **Release to PyPI**
2. Click **Run workflow**
3. Configure:
   - **Dry run**: ✅ Check this
   - **Python versions**: `3.11` (just one version)
   - **Targets**: `cuda` (just one target)
4. Click **Run workflow**

This builds only 1 wheel instead of 10, making it much faster to test.

### Method 4: Local Testing

Test the build process locally before using GitHub Actions.

**Steps:**
```bash
# Test building a single wheel locally
./build.sh cuda all 3.11

# Verify wheel was created
ls -lh wheelhouse-cuda/uccl-*.whl

# Validate the wheel
pip install twine
twine check wheelhouse-cuda/uccl-*.whl
```

## Testing Checklist

Before your first real release, test:

- [ ] **Dry run test** - Verify builds work
- [ ] **TestPyPI test** - Verify upload works
- [ ] **Wheel validation** - Verify wheels are valid
- [ ] **Artifact download** - Verify artifacts are created
- [ ] **Summary generation** - Verify summary is readable

## Step-by-Step First Test

### 1. Initial Dry Run Test

1. Go to GitHub repository → **Actions** tab
2. Select **Release to PyPI** workflow
3. Click **Run workflow** button (top right)
4. Set:
   - **Dry run**: ✅ **TRUE** (this is the key!)
   - **Python versions**: `3.11`
   - **Targets**: `cuda`
5. Click green **Run workflow** button
6. Wait for workflow to complete (may take 10-30 minutes)
7. Check:
   - ✅ All build steps succeeded
   - ✅ Validation passed
   - ✅ Artifacts were created
   - ✅ No upload step ran (check logs)

### 2. TestPyPI Test (Optional but Recommended)

1. Create TestPyPI account and token
2. Add `TESTPYPI_API_TOKEN` secret (or temporarily use `PYPI_API_TOKEN`)
3. Run workflow with:
   - **PyPI repository**: `testpypi`
   - **Dry run**: ❌ Unchecked
   - **Python versions**: `3.11`
   - **Targets**: `cuda`
4. Verify upload succeeded
5. Test installation from TestPyPI

### 3. Full Dry Run Test

Once basic tests pass, do a full dry run:

1. Run workflow with:
   - **Dry run**: ✅ TRUE
   - **Python versions**: `3.8 3.9 3.10 3.11 3.12`
   - **Targets**: `cuda rocm`
2. Verify all wheels build successfully
3. Download artifacts and verify all wheels are present

## Troubleshooting Tests

### Build Fails

- Check Docker setup in workflow logs
- Verify build.sh script works locally
- Check for architecture-specific issues

### Validation Fails

- Check wheel naming conventions
- Verify all required files are included
- Check for missing dependencies

### Upload Fails (TestPyPI)

- Verify API token is correct
- Check token has upload permissions
- Verify version doesn't already exist on TestPyPI

## Safety Tips

1. **Always use dry run first** for new configurations
2. **Use TestPyPI** before production PyPI
3. **Test with limited builds** to save time
4. **Check workflow logs** carefully before real release
5. **Verify artifacts** before trusting the workflow

## Production Release

Only after successful testing:

1. ✅ Dry run passed
2. ✅ TestPyPI test passed (optional but recommended)
3. ✅ All wheels validated
4. ✅ Ready for production

Then create a GitHub release to trigger the production workflow.

## Quick Reference

| Method | Safety | Speed | Completeness |
|--------|--------|-------|--------------|
| Dry Run | ✅✅✅ | Fast | Builds only |
| TestPyPI | ✅✅ | Medium | Full test |
| Limited Build | ✅✅✅ | Fastest | Partial |
| Local Test | ✅✅✅ | Fast | Builds only |

**Recommendation:** Start with Dry Run, then TestPyPI, then production.

