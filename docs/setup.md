# Setup

Requires Python 3.7+. To check your version of Python, run either:

```
python --version
python3 --version

```

Highly recommand setting up a virtual environment. The virtual environment lets you install packages that are only used for your assignments and do not impact the rest of the system. We suggest venv or anaconda.

For example, if you choose venv, run the following command:
```
python -m venv venv
source venv/bin/activate
```

The first line should be run only once, whereas the second needs to be run whenever you open a new terminal to get started for the class. You can tell if the second line works by checking if your terminal starts with (venv). See https://docs.python.org/3/library/venv.html for further instructions on how this works.


There are several external packages used. To install them in your virtual enviroment by running:

```
python -m pip install -r requirements.txt
conda install llvmlite
```

