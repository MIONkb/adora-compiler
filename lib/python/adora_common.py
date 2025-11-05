import os
import sys
import asyncio
from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn, TextColumn

class AdoraBase:
    def __init__(self, opts):
        self.opts = opts
        self.env = os.environ.copy()
        self.rootfolder = os.getcwd()
        self.ir_folder = os.path.join(self.rootfolder, "IR")
        self.temp_folder = os.path.join(self.ir_folder, "tempfiles")
        self.progress_bar = None

    def check_env(self, required_vars):
        """Check necessary env variables"""
        for var in required_vars:
            if var not in self.env:
                print(f"Error: {var} is not set. Please source your environment before running this script.")
                sys.exit(1)

    async def run_command(self, task, command, cwd=None):
        """Asyncio run commands"""
        command_str = " ".join(command)
        if task:
            self.progress_bar.update(task, advance=0, command=command_str[:30])
        proc = await asyncio.create_subprocess_exec(
            *command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self.env
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            print(f"Error: Command failed: {command_str}")
            print(stderr.decode())
            sys.exit(1)
        if task:
            self.progress_bar.update(task, advance=1, command="")
        return stdout.decode()

    def setup_progress_bar(self):
        """setup progress bar"""
        return Progress(
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[command]}")
        )