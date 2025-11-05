import os
import sys
import asyncio
import shutil
from argparse import ArgumentParser
from rich.progress import Progress, TimeElapsedColumn, MofNCompleteColumn, TextColumn

class AdoraNNCompiler:
    def __init__(self, opts):
        self.opts = opts
        self.env = os.environ.copy()
        self.rootfolder = os.getcwd()
        self.ir_folder = os.path.join(self.rootfolder, "IR")
        self.temp_folder = os.path.join(self.ir_folder, "tempfiles")
        self.progress_bar = None

    def check_env(self):
        """检查必要的环境变量是否已设置"""
        required_vars = ["CGRVOPT_PROJECT_PATH", "CGRA_ADG_PATH", "CHIPYARD_DIR"]
        for var in required_vars:
            if var not in self.env:
                print(f"Error: {var} is not set. Please source your environment before running this script.")
                sys.exit(1)

    async def run_command(self, task, command, cwd=None):
        """异步运行命令"""
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

    async def precompile_for_ai(self, task):
        """Stage 0：linalg to mlir"""
        kernel_src = self.opts.kernel
        ir_folder = self.ir_folder
        tar_folder = os.path.join(ir_folder, "0_kernels")
        os.makedirs(tar_folder, exist_ok=True)
        command = [
            "mlir-opt",
            "--allow-unregistered-dialect",
            "--affine-loop-normalize",
            "--affine-simplify-structures",
            "--normalize-memrefs",
            "--force-specialization",
            "--bufferization-bufferize",
            f"{kernel_src}.mlir",
            "-o", f"{ir_folder}/{kernel_src}_normalized.mlir"
        ]
        await self.run_command(task, command)

        command = [
            "cgra-opt",
            "--canonicalize",
            "-reconcile-unrealized-casts",
            "--affine-loop-fusion",
            "--adora-extract-affine-for-to-kernel",
            "--arith-expand", "--memref-expand",
            "-cse",
            f"--adora-extract-kernel-to-function=kernel-gen-dir={ir_folder}",
            f"{ir_folder}/{kernel_src}_normalized.mlir",
            "-o", f"{ir_folder}/{kernel_src}_host.mlir"
        ]
        await self.run_command(task, command)

    async def optimize_kernels(self, task):
        """阶段 1：优化内核"""
        src_folder = os.path.join(self.ir_folder, "0_kernels")
        tar_folder = os.path.join(self.ir_folder, "1_kernels_opt")
        os.makedirs(tar_folder, exist_ok=True)
        for file in os.listdir(src_folder):
            if file.endswith(".mlir"):
                filename = os.path.splitext(file)[0]
                command = [
                    "cgra-opt",
                    "--adora-simplify-loadstore",
                    "--adora-math-rewrite",
                    "--adora-adjust-kernel-mem-footprint=cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock",
                    os.path.join(src_folder, file),
                    "-o", os.path.join(tar_folder, f"{filename}_opt.mlir")
                ]
                await self.run_command(task, command)

    async def generate_dfgs(self, task):
        """阶段 2：生成 DFG"""
        src_folder = os.path.join(self.ir_folder, "1_kernels_opt")
        tar_folder = os.path.join(self.ir_folder, "2_dfgs")
        os.makedirs(tar_folder, exist_ok=True)
        for file in os.listdir(src_folder):
            if file.endswith("_opt.mlir"):
                filename = os.path.splitext(file)[0]
                command = [
                    "cgra-opt",
                    "--adora-kernel-dfg-gen",
                    os.path.join(src_folder, file)
                ]
                await self.run_command(task, command)
                dot_file = os.path.join(tar_folder, f"{filename}_CDFG.dot")
                png_file = os.path.join(tar_folder, f"{filename}_CDFG.png")
                await self.run_command(task, ["dot", dot_file, "-Tpng", "-o", png_file])

    async def map_kernels(self, task):
        """阶段 3：将 DFG 映射到 CGRA"""
        src_folder = os.path.join(self.ir_folder, "2_dfgs")
        tar_folder = os.path.join(self.ir_folder, "3_cgra_exes")
        os.makedirs(tar_folder, exist_ok=True)
        for file in os.listdir(src_folder):
            if file.endswith("_CDFG.dot"):
                filename = os.path.splitext(file)[0]
                command = [
                    "cgra-mapper",
                    f"--adg={self.env['CGRA_ADG_PATH']}/cgra_adg.json",
                    f"--op-file={self.env['CGRA_OP_FILE_PATH']}/operations.json",
                    f"--output={tar_folder}/{filename}_exe.c",
                    os.path.join(src_folder, file)
                ]
                await self.run_command(task, command)

    async def run_flow(self):
        """运行整个编译流程"""
        with Progress(
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[command]}")
        ) as progress:
            self.progress_bar = progress
            task = progress.add_task("[green] Compilation Flow", total=4, command="Starting")
            await self.precompile_for_ai(task)
            await self.optimize_kernels(task)
            await self.generate_dfgs(task)
            await self.map_kernels(task)
            progress.update(task, advance=1, command="Completed")

def parse_args():
    """解析命令行参数"""
    parser = ArgumentParser(description="Adora NN Compiler")
    parser.add_argument("--kernel", type=str, required=True, help="Kernel source file (without extension)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    opts = parse_args()
    compiler = AdoraNNCompiler(opts)
    compiler.check_env()
    asyncio.run(compiler.run_flow())

if __name__ == "__main__":
    main()