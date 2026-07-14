# KAUST Economy-Nature (KEN) model: A generalizable economic-biophysical SFC-IO framework
## Public releases of the KAUST Economy-Nature (KEN) model, developed and maintained at the KAUST Systems Science Lab

## Remark on current version
This version of the KEN model corresponds to the KEN model description and first preprint, which is forthcoming on SSRN:
“Transformation dynamics of an energy-rich and water-scarce economy in the Middle East”.

## Content:
- KEN Model: Data, code, visualizations, and run files.

## Documentation
See the paper:
Miess, M., Ilyas, A., Wang, D., Foramitti, F., Naqvi, A., Wada, Y. (2026, under review) Transformation dynamics of an energy-rich and water-scarce economy in the Middle East. 
For preprint on SSRN, see this link: (preprint link forthcoming).

## To prepare your local system:

- Install Python >= 3.12 from [python.org](https://www.python.org/downloads/) or [Anaconda](https://www.anaconda.com/download)
- Install requirements with `pip install -r requirements.txt`
- Install an IDE that supports Jupyter Notebooks such as [jupyter lab](https://jupyter.org/) or [VS Code](#vscode-tips) (see [VS Code Tips](#vscode-tips))

## To use the model:
- SINGLE scenario:     Run the model in `run_model_SINGLE_scenario.ipynb` (One single scenario one, many detailed analyses)
- SCENARIO COMPARISON: Run the model in `run_model_COMPARE_scenarios.ipynb` (Runs to compare all three scenarios, comparative graphs)

## Learning material:

- vscode (see [VS Code Tips](#vscode-tips))
- python https://docs.python.org/3/tutorial/index.html
- jupyter notebooks https://docs.jupyter.org/en/latest/
- numpy https://numpy.org/doc/stable/
- pandas https://pandas.pydata.org/docs/
- scipy fsolve https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html

## Model structure

Overview of the calculation order and dependencies of each variable.
![SFC_CORE Model Structure](KEN_framework.png)

## VSCode Tips

Install VSCode: https://code.visualstudio.com/

Recommended extensions to install: jupyter, pylance, prettier

How to use Jupyter notebooks in VSCode: https://code.visualstudio.com/docs/datascience/jupyter-notebooks

Recommendations for [`settings.json`](https://code.visualstudio.com/docs/getstarted/settings#_settingsjson)

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.autopep8",
    "editor.formatOnSave": true
  },
  "python.analysis.typeCheckingMode": "basic",
  "notebook.defaultFormatter": "ms-python.autopep8",
  "notebook.formatOnSave.enabled": true
}
```
