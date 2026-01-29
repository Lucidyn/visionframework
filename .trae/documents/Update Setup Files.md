Based on my analysis, the following updates are needed for the setup files:

## 1. Version Number Updates
- **setup.py**: Update version from 0.2.12 to 0.2.13 to match the README.md
- **pyproject.toml**: Update version from 0.2.12 to 0.2.13 to match the README.md

## 2. Dependency Consistency
- **pyproject.toml**: Add missing `huggingface_hub>=0.14.0` dependency to match setup.py and requirements.txt

## 3. Test Directory Path
- **pyproject.toml**: Update `testpaths` from ["tests"] to ["test"] to match the actual directory structure

## 4. File-by-File Changes

### setup.py
- Change line 10: `version="0.2.12"` → `version="0.2.13"`

### pyproject.toml
- Change line 7: `version = "0.2.12"` → `version = "0.2.13"`
- Add `"huggingface_hub>=0.14.0",` to the dependencies list (after line 38)
- Change line 85: `testpaths = ["tests"]` → `testpaths = ["test"]`

### requirements.txt
- No changes needed as it already includes all necessary dependencies

These updates will ensure version consistency across all files, fix dependency mismatches, and correct the test directory path configuration.