Write-Output "Installing pip packages one by one..."

# List of packages to install sequentially
$packages = @("gym==0.20.0", "unityagents==0.4.0")

# Install each package separately
foreach ($package in $packages) {
    Write-Output "Installing $package..."
    pip install $package
}

Write-Output "All pip packages installed successfully!"