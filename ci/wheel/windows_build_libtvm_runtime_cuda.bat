@echo off
rem Licensed to the Apache Software Foundation (ASF) under one
rem or more contributor license agreements.  See the NOTICE file
rem distributed with this work for additional information
rem regarding copyright ownership.  The ASF licenses this file
rem to you under the Apache License, Version 2.0 (the
rem "License"); you may not use this file except in compliance
rem with the License.  You may obtain a copy of the License at
rem
rem   http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing,
rem software distributed under the License is distributed on an
rem "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
rem KIND, either express or implied.  See the License for the
rem specific language governing permissions and limitations
rem under the License.
rem
rem Build tvm_runtime_cuda.dll on a Windows runner, run by the build_cuda_runtime
rem CI job (on the host; unlike Linux there is no build container on Windows).
rem Installs the pinned CUDA toolkit via conda and builds the sidecar into
rem build-wheel-cuda\lib\ for the wheel build to bundle. Windows mirror of
rem manylinux_build_libtvm_runtime_cuda.sh. Run with: shell: cmd.
setlocal enableextensions

rem repo root = this script's directory / ..\.. (native Windows paths; no cygpath).
pushd "%~dp0..\.." || exit /b 1
set "repo_root=%CD%"
popd
set "build_dir=%repo_root%\build-wheel-cuda"
set "cuda_prefix=C:\opt\cuda"

rem Locate conda: the runner ships Miniconda (exposed via %CONDA%) but it may not be
rem on PATH in this shell.
set "conda_exe=conda"
where conda >nul 2>nul || set "conda_exe=%CONDA%\Scripts\conda.exe"

rem Install the pinned CUDA toolkit via conda from the nvidia channel, mirroring the
rem LLVM-via-conda install used elsewhere. The win-64 channel caps at 13.0.x, so this
rem pins 13.0.2 -- slightly behind the Linux image's CUDA 13.1, which is harmless: the
rem sidecar has no device code and links the CUDA runtime by soname only. The nvidia
rem CDN occasionally returns a transient HTTP 5xx, so retry once; a half-finished first
rem attempt can leave the prefix partially populated, so wipe it before retrying.
if not exist "%cuda_prefix%\Library\bin\nvcc.exe" (
  call "%conda_exe%" create -q -p "%cuda_prefix%" -c nvidia/label/cuda-13.0.2 cuda-toolkit -y
  if errorlevel 1 (
    if exist "%cuda_prefix%" rmdir /s /q "%cuda_prefix%"
    call "%conda_exe%" create -q -p "%cuda_prefix%" -c nvidia/label/cuda-13.0.2 cuda-toolkit -y || exit /b 1
  )
)

rem conda lays the Windows toolkit out under <prefix>\Library (bin\nvcc.exe,
rem lib\x64\cudart.lib, include\...). Discover the root from nvcc.exe so TVM's
rem FindCUDA MSVC branch resolves against the real layout instead of a hardcode.
set "nvcc_exe="
for /f "delims=" %%i in ('dir /s /b "%cuda_prefix%\nvcc.exe" 2^>nul') do if not defined nvcc_exe set "nvcc_exe=%%i"
if not defined nvcc_exe ( echo nvcc.exe not found under %cuda_prefix% & exit /b 1 )
rem cuda_root = dirname(dirname(nvcc)) = <prefix>\Library
for %%i in ("%nvcc_exe%") do set "nvcc_bin=%%~dpi"
pushd "%nvcc_bin%.." || exit /b 1
set "cuda_root=%CD%"
popd
set "CUDA_PATH=%cuda_root%"

python -m pip install -U pip cmake ninja || exit /b 1
"%nvcc_exe%" --version || exit /b 1

rem nvcc needs the MSVC host compiler (cl.exe), so locate VS via vswhere and run the
rem cmake configure+build inside vcvars64 (this shell is not a VS Developer prompt).
set "vswhere=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
set "vs_path="
for /f "usebackq delims=" %%i in (`"%vswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "vs_path=%%i"
if not defined vs_path ( echo Visual Studio with VC tools not found & exit /b 1 )
set "vcvars=%vs_path%\VC\Auxiliary\Build\vcvars64.bat"

if exist "%build_dir%" rmdir /s /q "%build_dir%"

rem CMAKE_CUDA_COMPILER only tells CMake which nvcc to use (load-bearing: the conda
rem nvcc is not on PATH); it does not affect the resulting tvm_runtime_cuda.dll, which
rem is built only from .cc host sources (no .cu device code). CMAKE_CUDA_ARCHITECTURES
rem is intentionally not set -- a no-op for the same reason, and modern CMake fills a
rem default. -allow-unsupported-compiler guards against the runner's MSVC being newer
rem than the CUDA toolkit officially supports. The cmake command is kept on one line:
rem `^` continuations in a batch file break on any trailing whitespace.
rem CMake parses backslashes in string values as escapes (e.g. C:\opt -> invalid \o),
rem so hand cmake forward-slash paths. cmd builtins (rmdir / if exist) keep backslashes.
set "repo_root_fwd=%repo_root:\=/%"
set "build_dir_fwd=%build_dir:\=/%"
set "cuda_root_fwd=%cuda_root:\=/%"
set "nvcc_fwd=%nvcc_exe:\=/%"

call "%vcvars%" || exit /b 1
cmake -S "%repo_root_fwd%/3rdparty/tvm-tirx" -B "%build_dir_fwd%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DTVM_TIRX_BUILD_TESTS=OFF -DTVM_TIRX_BUILD_PYTHON_MODULE=ON -DTVM_TIRX_USE_CUDA="%cuda_root_fwd%" -DTVM_TIRX_USE_LLVM=OFF -DCMAKE_CUDA_COMPILER="%nvcc_fwd%" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" || exit /b 1
cmake --build "%build_dir_fwd%" --target tvm_runtime tvm_runtime_cuda --config Release || exit /b 1

if not exist "%build_dir%\lib\tvm_runtime_cuda.dll" ( echo tvm_runtime_cuda.dll was not produced & exit /b 1 )
rem No patchelf/rpath step on Windows; delvewheel vendors dependencies at repair time.
echo CUDA runtime: %build_dir%\lib\tvm_runtime_cuda.dll
endlocal
