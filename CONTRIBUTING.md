# Contributing

1. Fork the repository or create a feature branch from `main`.
2. Install Python 3.12 and run `python -m pip install -r requirements.txt`.
3. Make a focused change and add or update tests for its behavior.
4. Run `python -m pytest -q`.
5. Push the branch and open a pull request against `main` with a short description of the change and its test result.

The CI workflow runs for pull requests. Changes must pass CI and be reviewed before merging. Do not commit `.env`, databases, student information, face images, or downloaded face-model weights.