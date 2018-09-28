#!/usr/bin/env bash
#
# Script to install Docker with NVIDIA CUDA capabilities for Deep Learning on an AWS EC2 instance.
#
# Note that this script only supports P2 instances with Ubuntu 16.04 as the architecture and versio numbers are
# matched to this instance. See also:
#
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html
#

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo systemctl status docker
sudo usermod -a -G docker ubuntu

# Install CUDA drivers for AWS P2 instances
wget http://us.download.nvidia.com/tesla/384.145/nvidia-diag-driver-local-repo-ubuntu1604-384.145_1.0-1_amd64.deb
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-384.145_1.0-1_amd64.deb
sudo apt-key add /var/nvidia-diag-driver-local-repo-384.145/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda-drivers

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo systemctl restart docker

# Test if installation went fine using nvidia-smi or nvcc using the latest official CUDA image (OPTIONAL)
# sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
# sudo docker run --runtime=nvidia --rm nvidia/cuda nvcc --version
