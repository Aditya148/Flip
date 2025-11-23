# Publishing to PyPI

This guide explains how to build and publish the Flip SDK to PyPI.

## Prerequisites

1. **Install build tools**:
```bash
pip install build twine
```

2. **Create PyPI account**:
   - Go to https://pypi.org/account/register/
   - Verify your email
   - Set up 2FA (recommended)

3. **Create API token**:
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Save it securely

## Building the Package

### 1. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info
```

### 2. Build the Package

```bash
# Build source distribution and wheel
python -m build
```

This creates:
- `dist/flip-rag-0.1.0.tar.gz` (source distribution)
- `dist/flip_rag-0.1.0-py3-none-any.whl` (wheel)

### 3. Check the Build

```bash
# Check package metadata
twine check dist/*
```

## Testing on TestPyPI (Recommended)

Before publishing to PyPI, test on TestPyPI:

### 1. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your TestPyPI API token

### 2. Test Installation

```bash
# Create a new virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ flip-rag

# Test it
python -c "from flip import Flip; print('Success!')"
```

## Publishing to PyPI

Once tested, publish to the real PyPI:

```bash
twine upload dist/*
```

You'll be prompted for:
- Username: `__token__`
- Password: Your PyPI API token

## Using .pypirc (Optional)

Create `~/.pypirc` to avoid entering credentials each time:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**Important**: Keep this file secure! Add it to `.gitignore`.

## Version Management

### Updating Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.1.1"  # Increment version
```

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `0.1.0` → `0.1.1` (bug fixes)
- `0.1.0` → `0.2.0` (new features)
- `0.1.0` → `1.0.0` (breaking changes)

## Complete Workflow

```bash
# 1. Update version in pyproject.toml
# 2. Clean old builds
rm -rf dist/ build/ *.egg-info

# 3. Build
python -m build

# 4. Check
twine check dist/*

# 5. Test on TestPyPI (optional but recommended)
twine upload --repository testpypi dist/*

# 6. Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ flip-rag

# 7. If all good, upload to PyPI
twine upload dist/*

# 8. Verify
pip install flip-rag
```

## After Publishing

### Installation

Users can now install with:

```bash
pip install flip-rag
```

### With Optional Dependencies

```bash
# Development tools
pip install flip-rag[dev]

# Additional vector stores
pip install flip-rag[pinecone]
pip install flip-rag[qdrant]
pip install flip-rag[weaviate]

# All optional dependencies
pip install flip-rag[all]
```

## Troubleshooting

### Build Fails

If `python -m build` fails:

1. **Check pyproject.toml syntax**:
   ```bash
   python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"
   ```

2. **Ensure all files exist**:
   - `README.md`
   - `LICENSE`
   - `flip/__init__.py`

3. **Remove setup.py** (if it exists):
   ```bash
   rm setup.py
   ```

### Upload Fails

If `twine upload` fails:

1. **Check credentials**: Ensure API token is correct
2. **Check version**: Version must be unique (can't re-upload same version)
3. **Check package name**: Name must be available on PyPI

### Package Name Already Taken

If `flip-rag` is taken, choose a different name:

1. Edit `pyproject.toml`:
   ```toml
   [project]
   name = "flip-rag-sdk"  # or another name
   ```

2. Rebuild and upload

## Best Practices

1. **Always test on TestPyPI first**
2. **Use API tokens, not passwords**
3. **Keep .pypirc secure**
4. **Increment version for each release**
5. **Write clear release notes**
6. **Tag releases in git**:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

## Continuous Deployment (Optional)

### GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Add `PYPI_API_TOKEN` to GitHub repository secrets.

## Resources

- [PyPI](https://pypi.org/)
- [TestPyPI](https://test.pypi.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
