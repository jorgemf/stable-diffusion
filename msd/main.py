import typer
from .dataset import app as dataset_app
from .train import train
from .test import test
from .config import setup

app = typer.Typer()
app.add_typer(dataset_app, name="dataset")
app.command()(train)
app.command()(test)

if __name__ == '__main__':

    # import subprocess
    # command = "nvidia-smi"
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print(result.stdout)

    setup()
    app()
