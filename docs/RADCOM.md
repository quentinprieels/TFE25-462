# Using the RADCOM library to run the python scripts

The RADCOM library is developed by François De Saint Moulin at Université Catholique de Louvain. To use the library and work with a proper environment, you need to follow the steps below.

## Step 1: Create a new python environment

First, you need to create a new python environment. In this project, we assume that you will locate the environment in the root directory of this repository.
You can do this by running the following command in your terminal:

```bash
python3 -m venv env
```

Then, you need to activate the environment by running the following command:

```bash
source env/bin/activate
```

## Step 2: Install the required packages

Before installing the required packages, you need to modify the `requirements.txt` file to make shure that the specified path to the RADCOM library is correct. You can do this by changing the path in the `requirements.txt` file to the path where the RADCOM library is located on your machine.

```bash
-e /path/to/RADCOM
```

After modifying the `requirements.txt` file, you can install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

In this way, the RADCOM library can be modified and used in the project.
