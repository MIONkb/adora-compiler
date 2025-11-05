#!/usr/bin/env python3

import asyncio
from argparse import ArgumentParser
from pypack.adora_common import AdoraBase

class AdoraNNCompiler(AdoraBase):
    async def precompile_for_ai(self, task):
        """Stage 0: Precompile for AI."""
        kernel_src = self.opts.kernel
        ir_folder = self.ir_folder
        tar_folder = os.path.join(ir_folder, "0_kernels")
        os.makedirs(tar_folder, exist_ok=True)

        # Command to normalize and precompile the MLIR file
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

        # Command to extract kernels
        command = [
            "cgra-opt",
            "--canonicalize",
            "--reconcile-unrealized-casts",
            "--affine-loop-fusion",
            "--adora-extract-affine-for-to-kernel",
            "--arith-expand",
            "--memref-expand",
            "--cse",
            f"--adora-extract-kernel-to-function=kernel-gen-dir={tar_folder}",
            f"{ir_folder}/{kernel_src}_normalized.mlir",
            "-o", f"{ir_folder}/{kernel_src}_host.mlir"
        ]
        await self.run_command(task, command)

    async def optimize_kernels(self, task):
        """Stage 1: Optimize kernels."""
        src_folder = os.path.join(self.ir_folder, "0_kernels")
        tar_folder = os.path.join(self.ir_folder, "1_kernels_opt")
        os.makedirs(tar_folder, exist_ok=True)

        # Traverse all MLIR files in the source folder
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
        """Stage 2: Generate DFGs."""
        src_folder = os.path.join(self.ir_folder, "1_kernels_opt")
        tar_folder = os.path.join(self.ir_folder, "2_dfgs")
        os.makedirs(tar_folder, exist_ok=True)

        # Traverse all optimized MLIR files in the source folder
        for file in os.listdir(src_folder):
            if file.endswith("_opt.mlir"):
                filename = os.path.splitext(file)[0]
                command = [
                    "cgra-opt",
                    "--adora-kernel-dfg-gen",
                    os.path.join(src_folder, file)
                ]
                await self.run_command(task, command)

                # Convert the generated DFG to a PNG image
                dot_file = os.path.join(tar_folder, f"{filename}_CDFG.dot")
                png_file = os.path.join(tar_folder, f"{filename}_CDFG.png")
                await self.run_command(task, ["dot", dot_file, "-Tpng", "-o", png_file])

    async def map_kernels(self, task):
        """Stage 3: Map kernels to CGRA."""
        src_folder = os.path.join(self.ir_folder, "2_dfgs")
        tar_folder = os.path.join(self.ir_folder, "3_cgra_exes")
        os.makedirs(tar_folder, exist_ok=True)

        # Traverse all DFG files in the source folder
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
        """Run the entire compilation flow."""
        with self.setup_progress_bar() as progress:
            self.progress_bar = progress
            task = progress.add_task("[green] Compilation Flow", total=4, command="Starting")
            await self.precompile_for_ai(task)
            await self.optimize_kernels(task)
            await self.generate_dfgs(task)
            await self.map_kernels(task)
            progress.update(task, advance=1, command="Completed")

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Adora Neural Network Compiler")
    parser.add_argument(
        "mlir_file",
        type=str,
        help="mlir source file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def main():
    opts = parse_args()
    compiler = AdoraNNCompiler(opts)
    compiler.check_env(["CGRVOPT_PROJECT_PATH", "CGRA_ADG_PATH", "CHIPYARD_DIR"])
    asyncio.run(compiler.run_flow())

if __name__ == "__main__":
    main()