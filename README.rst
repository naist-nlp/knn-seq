kNN-MT
######

Installation
============

.. code:: bash

    git clone https://github.com/naist-nlp/knn-seq.git
    cd knn-seq/
    pip install ./

Usage
=====

First, pre-process the dataset for building the datastore.

.. code:: bash

    NUM_WORKERS=16  # specify the number of CPUs
    DATABIN_DIR=binarized
    INDEX_DIR=${DATABIN_DIR}/index/en  # index directory must be `${binarized_data}/index/${tgt_lang}`

    # Pre-processing the validation/test set.
    fairseq-preprocess \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --validpref corpus/valid \
        --testpref corpus/test \
        --destdir ${DATABIN_DIR} \
        --workers ${NUM_WORKERS}

    # Pre-processing the corpus that is used as datastore.
    python knn_seq/cli/binarize_fairseq.py \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --trainpref corpus/datastore-text \
        --workers ${NUM_WORKERS} \
        ${INDEX_DIR}

Next, create the datastore by computing all key vectors.

.. code:: bash

    python knn_seq/cli/create_datastore_fairseq.py \
        --user-dir knn_seq/ \
        --task translation_knn \
        --knn-key ffn_in \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --save-freq 512 \
        --max-tokens 16384 \
        --num-workers ${NUM_WORKERS} \
        --fp16 \
        ${INDEX_DIR}

Then, build the index for efficient kNN search.

.. code:: bash

    python knn_seq/cli/build_index.py \
        -d ${INDEX_DIR} \
        --train-size 5242880 \
        --chunk-size 10000000 \
        --feature ffn_in \
        --metric l2 \
        --hnsw-edges 32 \  # Coarse quantizer to search nearest top-`nprobe` centroids
        --ivf-lists 131072 \  # K-means clustering
        --pq-subvec 64 \  # Product quantization (PQ) to compress the all vectors to uint8 codes.
        --use-opq \  # Rotaion vectors to minimize the
        --safe \
        --verbose

- :code:`--hnsw-edges`: HNSW is used as coarse quantizer to search nearest top-`nprobe` centroids.
  This option specifies the number of edges in construction HNSW graph.
- :code:`--ivf-lists`: IVF (inverted file index) does k-means clustering for faster search.
  This option specifies the number of clusters in k-means.
- :code:`--pq-subvec`: PQ (product quantization) splits a vector to M sub-spaces and quantizes in each sub-space.
  This option specifies the number of sub-spaces (:code:`M`).
- :code:`--use-opq`: OPQ (Optimized PQ) rotates raw vectors to minimize the quantization error. It can also reduce the dimension size.

The following option can be specified:

- :code:`--use-pca`: PCA reduces the dimension of vectors.
- :code:`--transform-dim`: Reduced dimension size in OPQ or PCA.


Last, generate sentences with kNN.

.. code:: bash

    fairseq-generate \
        --user-dir knn_seq/ \
        --task translation_knn \
        --fp16 \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --gen-subset test \
        --beam 5 \
        --knn-key ffn_in \
        --knn-metric l2 \
        --knn-topk 16 \  # The number of nearest neighbors.
        --knn-nprobe 64 \ # The number of nearest centroids for IVF search.
        --knn-temperature 100.0 \  # Temperature of kNN softmax.
        --knn-weight 0.5 \  # kNN-MT interpolation parameter.
        --knn-fp16 \
        ${DATABIN_DIR}
