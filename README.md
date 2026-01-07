mol_platform

This repository is an archived snapshot of the mol_platform codebase, preserved for long-term reference and reproducibility.

The platform requires Conda for environment management. Please ensure Conda is installed before use.

▶ Launch (assume the current directory is the repository root)

- Start with environment initialization:
  python start.py

- Or start the Streamlit app directly (requires manual dependency installation; see environment.md):
  streamlit run result_analysis/app.py

▶ Model Upload and Usage

The platform supports uploading model files. Model execution may require a separately configured runtime environment.
Please refer to env.md or requirements.txt for environment setup details.

You may use the built-in environment management utility:
  python env_utils.py -h

▶ Refresh

Use Ctrl+R to refresh the page during development/debugging.
Because the platform reads/writes local files and interacts with the local environment, Streamlit may display a "File Change" prompt.
In that case, manually refresh the page or restart the process.

▶ Shutdown

Use the in-app "Close" control, or terminate the process from the terminal.

▶ Template Packages

This archive references template packages:
- moleculenet_model.zip
- moleculenet_data.zip

These files were distributed via an external link with a limited validity period. If the link has expired or you do not have access, please contact the original maintainers for an updated source.

Attribution:
Zhejiang University Data Science Research Center
Miao Lab, Zhejiang University