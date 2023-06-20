knn-seq
#######

Installation
============

.. code:: bash

    git clone https://github.com/naist-nlp/knn-seq.git
    cd knn-seq/
    pip install ./

Usage
=====

kNN-MT (Khandelwal et al., 2021)
--------------------------------

First, preprocess the dataset for building the datastore.

.. code:: bash

    NUM_WORKERS=16  # specify the number of CPUs
    DATABIN_DIR=binarized
    INDEX_DIR=${DATABIN_DIR}/index/en  # index directory must be `${binarized_data}/index/${tgt_lang}`

    # Preprocess the validation/test set.
    fairseq-preprocess \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --validpref corpus/valid \
        --testpref corpus/test \
        --destdir ${DATABIN_DIR} \
        --workers ${NUM_WORKERS}

    # Preprocess the corpus that is used as datastore.
    python knn_seq/cli/binarize_fairseq.py \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --trainpref corpus/datastore-text \
        --workers ${NUM_WORKERS} \
        ${INDEX_DIR}

Next, construct the datastore by computing all key vectors.

.. code:: bash

    python knn_seq/cli/create_datastore_fairseq.py \
        --knn-key ffn_in \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
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
        --use-opq \  # Rotaion vectors to minimize the PQ error.
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
        --knn-key ffn_in \
        --knn-metric l2 \
        --knn-topk 64 \  # The number of nearest neighbors.
        --knn-nprobe 32 \ # The number of nearest centroids for IVF search.
        --knn-temperature 100.0 \  # Temperature of kNN softmax.
        --knn-weight 0.5 \  # kNN-MT interpolation parameter.
        ${DATABIN_DIR}

________

Subset kNN-MT (Deguchi et al., 2023)
------------------------------------

The process is the same as in naive kNN-MT up to the target key vector computation using :code:`create_dastore_fairseq.py`.

Subset kNN-MT quantizes the target key vectors instead of building the kNN index.

.. code:: bash

    python knn_seq/cli/build_index.py \
        -d ${INDEX_DIR} \
        --outpref pq \
        --train-size 5242880 \
        --chunk-size 10000000 \
        --feature ffn_in \
        --metric l2 \
        --pq-subvec 64 \  # Product quantization (PQ) to compress the all vectors to uint8 codes.
        --use-pca \
        --transform-dim 256 \  # Reduce the dimension size by PCA
        --safe \
        --verbose

Next, construct the sentence datastore.

- Case1: Use LaBSE from sentence-transformers for the sentence encoder

.. code:: bash

    SRC_KEY=senttr
    SRC_INDEX_DIR=${DATABIN_DIR}/index/de.${SRC_KEY}  # source index directory must be `{binarized_data}/index/${src_lang}.{src_key}`

    # Preprocess the source text that is used for the sentence datastore.
    # In this case, give the detokenized source-side text. Sentences will be tokenized by the LaBSE tokenizer in :code:`binarize.py`.
    python knn_seq/cli/binarize.py \
        --input corpus/datastore-text.detok.de \
        --outdir ${SRC_INDEX_DIR} \
        sentence-transformers/LaBSE  # cf. https://huggingface.co/sentence-transformers/LaBSE

    # Construct the sentence datastore.
    python knn_seq/cli/create_datastore.py \
        --outdir ${SRC_INDEX_DIR} \
        --fp16 \
        --feature senttr \
        sentence-transformers/LaBSE


- Case2: Use an NMT encoder itself as the sentence encoder

.. code:: bash

    SRC_KEY=enc
    SRC_INDEX_DIR=${DATABIN_DIR}/index/de.${SRC_KEY}  # source index directory must be `{binarized_data}/index/${src_lang}.{src_key}`

    # Preprocess the source text that is used for the sentence datastore.
    python knn_seq/cli/binarize_fairseq.py \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --trainpref corpus/datastore-text \
        --workers ${NUM_WORKERS} \
        --binarize-src \  # Binarize the source text.
        ${SRC_INDEX_DIR}

    # Construct the sentence datastore.
    python knn_seq/cli/create_datastore_fairseq.py \
        --src-key ${SRC_KEY} \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --num-workers ${NUM_WORKERS} \
        --fp16 \
        --store-src-sent \
        ${SRC_INDEX_DIR}

Then, build the index of the sentence datastore.

.. code:: bash

    python knn_seq/cli/build_index.py \
        -d ${SRC_INDEX_DIR} \
        --train-size 5242880 \
        --chunk-size 10000000 \
        --feature ${SRC_KEY} \
        --metric l2 \
        --hnsw-edges 32 \  # Coarse quantizer to search nearest top-`nprobe` centroids
        --ivf-lists 32768 \  # K-means clustering
        --pq-subvec 64 \  # Product quantization (PQ) to compress the all vectors to uint8 codes.
        --use-opq \  # Rotaion vectors to minimize the PQ error.
        --transform-dim 256 \  # Reduce the dimension size.
        --safe \
        --verbose

Generate translations using subset kNN-MT.

.. code:: bash

   # Case1: sentence-tranformers/LaBSE
   # Copy the detokenized source sentence to query the neighbor sentences by LaBSE.
   fairseq-preprocess \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --testpref corpus/test \
        --destdir ${DATABIN_DIR}/orig \
        --dataset-impl raw  # Just copy the text files.

   # Generate.
   fairseq-generate \
        --user-dir knn_seq/ \
        --task translation_knn \
        --fp16 \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --knn-key ffn_in \
        --knn-metric l2 \
        --knn-topk 64 \  # The number of nearest neighbors.
        --knn-nprobe 32 \ # The number of nearest centroids for IVF search.
        --knn-temperature 100.0 \  # Temperature of kNN softmax.
        --knn-weight 0.5 \  # kNN-MT interpolation parameter.
        --src-key ${SRC_KEY} \
        --src-metric l2 \
        --src-knn-model sentence-transformers/LaBSE \
        --src-topk 512 \  # Search for the 512 nearest neighbor sentences of the input.
        --src-nprobe 64 \
        --src-efsearch 64 \
        ${DATABIN_DIR}

   # Case2: NMT encoder
   # Generate.
   fairseq-generate \
        --user-dir knn_seq/ \
        --task translation_knn \
        --fp16 \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --knn-key ffn_in \
        --knn-metric l2 \
        --knn-topk 64 \  # The number of nearest neighbors.
        --knn-nprobe 32 \ # The number of nearest centroids for IVF search.
        --knn-temperature 100.0 \  # Temperature of kNN softmax.
        --knn-weight 0.5 \  # kNN-MT interpolation parameter.
        --src-key ${SRC_KEY} \
        --src-metric l2 \
        --src-topk 512 \  # Search for the 512 nearest neighbor sentences of the input.
        --src-nprobe 64 \
        --src-efsearch 64 \
        ${DATABIN_DIR}
