# pull ML perf
git clone https://github.com/mlcommons/inference.git

#get imagenet dataset
mkdir data
cd data/
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir val
mv ILSVRC2012_img_val.tar val
cd val
tar -xvf ILSVRC2012_img_val.tar 
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
rm ILSVRC2012_img_val.tar 