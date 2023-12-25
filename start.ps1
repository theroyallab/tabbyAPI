# Arg parsing
param(
    [switch]$ignore_upgrade = $false,
    [switch]$activate_venv = $false
)

# Gets the currently installed CUDA version
function GetRequirementsFile {
    $GpuInfo = (Get-WmiObject Win32_VideoController).Name
    if ($GpuInfo.Contains("AMD")) {
        Write-Output "AMD/ROCm isn't supported on Windows. Please switch to linux."
        exit
    }

    $CudaPath = $env:CUDA_PATH
    $CudaVersion = Split-Path $CudaPath -Leaf

    # Decide requirements based on CUDA version
    if ($CudaVersion.Contains("12")) {
        return "requirements"
    } elseif ($CudaVersion.Contains("11.8")) {
        return "requirements-cu118"
    } else {
        Write-Output "Script cannot find your CUDA installation. installing from requirements-nowheel.txt"
        return "requirements-nowheel"
    }
}

# Make a venv and enter it
function CreateAndActivateVenv {
    # Is the user using conda?
    if ($null -ne $env:CONDA_PREFIX) {
        Write-Output "It looks like you're in a conda environment. Skipping venv check."
        return
    }

    $VenvDir = "$PSScriptRoot\venv"

    if (!(Test-Path -Path $VenvDir)) {
        Write-Output "Venv doesn't exist! Creating one for you."
        python -m venv venv
    }

    . "$VenvDir\Scripts\activate.ps1"

    if ($activate_venv) {
        Write-Output "Stopping at venv activation due to user request."
        exit
    }
}

# Entrypoint for API start
function StartAPI {
    pip -V
    if ($ignore_upgrade) {
        Write-Output "Ignoring pip dependency upgrade due to user request."
    } else {
        pip install --upgrade -r "$RequirementsFile.txt"
    }

    python main.py
}

# Navigate to the script directory
Set-Location $PSScriptRoot
$RequirementsFile = GetRequirementsFile
CreateAndActivateVenv
StartAPI
