#! /bin/bash

# get just split files needed for training

echo "getting snapshot data.."
echo "if this fails, run aws configure"
cd ~
mkdir -p deeptrade-data/
aws s3 sync s3://deeptrade.data/candles/ deeptrade-data/candles/
aws s3 sync s3://deeptrade.data/gdax_book/BTC-USD/snapshots/ deeptrade-data/gdax_book/BTC-USD/snapshots
# uncomment for more data
#aws s3 sync s3://deeptrade.data/gdax_book/BTC-USD/split/ deeptrade-data/gdax_book/BTC-USD/split

# setup envs
# Recommended: using EC2 deep learning AMI and conda:
pip uninstall tensorflow
pip install tensorflow-gpu==1.12

# ubuntu
sudo -y apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
sudo -y apt-get install mpich build-essential

# osx: brew install cmake openmpi
# install ta-lib (needed for python wrapper TA-lib)
# see https://github.com/mrjbq7/ta-lib#dependencies
cd ~
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

cd deeptrade2
pip install -r requirements.txt

# add these to `.bashrc`
echo "export DEEPTRADE_DATA=/home/ubuntu/deeptrade-data/" >> ~/.bashrc
echo "export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'" >> ~/.bashrc
echo "export OPENAI_LOGDIR=/home/ubuntu/deeptrade-logs/" >> ~/.bashrc
