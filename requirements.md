# Requirements
## Cuda
sudo apt install nvidia-cuda-toolkit
nvcc --version

## Numpy, Pytorch, Sexpdata, Bvh
sudo apt install python3-pip
pip3 install numpy
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-linux_x86_64.whls
pip3 install torchvision
pip3 install sexpdata
pip3 install bvh

## Sublime
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
sudo apt-add-repository "deb https://download.sublimetext.com/ apt/stable/"
sudo apt install sublime-text

## Git
sudo apt-get install git

## Simspark
sudo apt-get install g++ subversion cmake libfreetype6-dev libode-dev libsdl-dev ruby ruby-dev libdevil-dev libboost-dev libboost-thread-dev libboost-regex-dev libboost-system-dev qt4-default
svn co https://svn.code.sf.net/p/simspark/svn/trunk simspark

### Spark
cd simspark/spark
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig

### Server
cd simspark/rcssserver3d
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig

### This repo
https://github.com/rishshah/RoboSoccer.git
Modify ld_library_path, paths in env.py
