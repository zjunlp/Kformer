echo "----------------- Install Python Package -----------------"
# pip install --user allennlp
# sudo chmod -R /opt/conda/bin/f2py
# echo y | pip uninstall numpy
# pip uninstall spacy
pip install --user tensorboardX six tqdm scikit-learn lmdb pyarrow py-lz4framed methodtools pathlib
cd $PWD_DIR

pip install --user --editable .
pip install --user --upgrade numpy
ulimit -SHn 51200