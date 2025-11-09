#!/usr/bin/env python3
import os
import sys
import asyncio
from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn, TextColumn

class AdoraBase:
    def __init__(self, opts):
        """
        Common base for Adora compilers.

        - opts: argparse.Namespace from parse_args()
          Expected possible fields: mlir_file (positional), kernel (optional), verbose (optional)
        This constructor will normalize fields:
        - opts.mlir_path: absolute path to mlir file if provided
        - opts.kernel: basename without extension (derived from mlir_file if not provided)
        - self.rootfolder: directory containing mlir_file or current working directory
        """
        self.opts = opts
        # normalize mlir / kernel fields
        if hasattr(self.opts, "mlir_file") and self.opts.mlir_file:
            mlir_path = os.path.abspath(self.opts.mlir_file)
            self.opts.mlir_path = mlir_path
            # if kernel not provided, derive from mlir filename
            if not hasattr(self.opts, "kernel") or not self.opts.kernel:
                self.opts.kernel = os.path.splitext(os.path.basename(mlir_path))[0]
            # set rootfolder to mlir file directory for stage outputs
            self.rootfolder = os.path.dirname(mlir_path)
        else:
            # fallback to cwd
            self.opts.mlir_path = None
            if not hasattr(self.opts, "kernel"):
                self.opts.kernel = None
            self.rootfolder = os.getcwd()

        # copy environment and allow overrides via opts if provided
        self.env = os.environ.copy()
        if hasattr(self.opts, "cgraf_adg_path") and self.opts.cgraf_adg_path:
            self.env["CGRA_ADG_PATH"] = self.opts.cgraf_adg_path
        if hasattr(self.opts, "cgraf_op_file_path") and self.opts.cgraf_op_file_path:
            self.env["CGRA_OP_FILE_PATH"] = self.opts.cgraf_op_file_path

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
        """
        Run an external command asynchronously.

        - command: list of program + args (recommended)
        - task: rich progress task (may be None)
        - cwd: working directory for the command
        """
        # Normalize command display
        if isinstance(command, (list, tuple)):
            command_str = " ".join(command)
        else:
            command_str = str(command)

        if getattr(self.opts, "verbose", False):
            print(f"[run_command] cwd={cwd or os.getcwd()} cmd={command_str}", file=sys.stderr)

        if task and self.progress_bar:
            # show a short preview of the command in progress bar
            self.progress_bar.update(task, advance=0, command=command_str[:60])

        # Prefer list form for create_subprocess_exec
        if isinstance(command, str):
            # run via shell if string provided
            proc = await asyncio.create_subprocess_shell(
                command, cwd=cwd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self.env
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *command, cwd=cwd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self.env
            )

        stdout, stderr = await proc.communicate()
        out = stdout.decode().strip()
        err = stderr.decode().strip()

        if proc.returncode != 0:
            print(f"Error: Command failed (rc={proc.returncode}): {command_str}", file=sys.stderr)
            if err:
                print(err, file=sys.stderr)
            sys.exit(proc.returncode)

        if task and self.progress_bar:
            self.progress_bar.update(task, advance=1, command="")

        if getattr(self.opts, "verbose", False) and out:
            print(out, file=sys.stderr)

        return out

    def setup_progress_bar(self):
        """setup progress bar"""
        return Progress(
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[command]}")
        )

    def make_ir_dirs(self):
        """Create IR layout directories used by nncompiler stages."""
        os.makedirs(self.ir_folder, exist_ok=True)
        os.makedirs(self.temp_folder, exist_ok=True)
        for d in ("0_kernels", "1_kernels_opt", "2_dfgs", "3_cgra_exes"):
            os.makedirs(os.path.join(self.ir_folder, d), exist_ok=True)