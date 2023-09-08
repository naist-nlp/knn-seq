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
    INDEX_DIR=${DATABIN_DIR}/index/en

    # Preprocess the validation/test set.
    fairseq-preprocess \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --validpref corpus/valid \
        --testpref corpus/test \
        --destdir ${DATABIN_DIR} \
        --workers ${NUM_WORKERS}

    # Store values of the datastore.
    python knn_seq/cli/store_values.py \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --trainpref corpus/datastore-text \
        --workers ${NUM_WORKERS} \
        ${INDEX_DIR}

Next, store all key vectors in a key storage.

.. code:: bash

    python knn_seq/cli/store_keys.py \
        --knn-key ffn_in \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --num-workers ${NUM_WORKERS} \
        --fp16 \
        --max-tokens 6000 \
        ${INDEX_DIR}

Then, build the key index for efficient kNN search.

.. code:: bash

    INDEX_PATH_PREFIX=${INDEX_DIR}/index.ffn_in.l2.M64.nlist131072

    python knn_seq/cli/build_index.py \
        --key-storage ${INDEX_DIR}/keys.ffn_in.bin \
        --index-path-prefix ${INDEX_PATH_PREFIX} \
        --train-size 5242880 \
        --chunk-size 10000000 \
        --metric l2 \
        --hnsw-edges 32 \  # Coarse quantizer to search nearest top-`nprobe` centroids
        --ivf-lists 131072 \  # K-means clustering
        --pq-subvec 64 \  # Product quantization (PQ) to compress the all vectors to uint8 codes.
        --use-opq  # Rotaion vectors to minimize the PQ error.

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
        --max-tokens 6000 \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --knn-index-path ${INDEX_PATH_PREFIX}.bin \
        --knn-value-path ${INDEX_DIR}/values.bin \
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

The process is the same as in naive kNN-MT up to the target key vector computation using :code:`store_keys.py`.

Subset kNN-MT quantizes the target key vectors instead of building the kNN index.

.. code:: bash

    PQ_PATH_PREFIX=${INDEX_DIR}/pq.ffn_in.M64

    python knn_seq/cli/build_index.py \
        --key-storage ${INDEX_DIR}/keys.ffn_in.bin \
        --index-path-prefix ${PQ_PATH_PREFIX} \
        --train-size 5242880 \
        --chunk-size 10000000 \
        --pq-subvec 64 \  # Product quantization (PQ) to compress the all vectors to uint8 codes.
        --use-pca \
        --transform-dim 256  # Reduce the dimension size by PCA

Next, store the sentence key vectors.

- Case1: Use LaBSE from sentence-transformers for the sentence encoder

.. code:: bash

    SRC_KEY=senttr
    SRC_INDEX_DIR=${DATABIN_DIR}/index/de.${SRC_KEY}
    SRC_INDEX_PATH_PREFIX=${SRC_INDEX_DIR}/index.${SRC_KEY}.l2.nlist32768.M64

    # Store values of the sentence datastore.
    # In this case, give the detokenized source-side text.
    # Sentences will be tokenized by the LaBSE tokenizer in :code:`store_values_hf.py`.
    python knn_seq/cli/store_values_hf.py \
        --input corpus/datastore-text.detok.de \ # Detokenized text
        --outdir ${SRC_INDEX_DIR} \
        sentence-transformers/LaBSE  # cf. https://huggingface.co/sentence-transformers/LaBSE

    # Store key vectors of the sentence datastore.
    python knn_seq/cli/store_keys_hf.py \
        --outdir ${SRC_INDEX_DIR} \
        --fp16 \
        --max-tokens 6000 \
        --feature senttr \
        sentence-transformers/LaBSE


- Case2: Use an NMT encoder itself as the sentence encoder

.. code:: bash

    SRC_KEY=enc
    SRC_INDEX_DIR=${DATABIN_DIR}/index/de.${SRC_KEY}  # source index directory must be `{binarized_data}/index/${src_lang}.{src_key}`

    # Store values of the sentence datastore.
    python knn_seq/cli/store_values.py \
        --source-lang de --target-lang en \
        --srcdict wmt19.de-en.ffn8192/dict.de.txt \
        --tgtdict wmt19.de-en.ffn8192/dict.en.txt \
        --trainpref corpus/datastore-text \  # Tokenized text
        --workers ${NUM_WORKERS} \
        --binarize-src \  # Binarize the source text.
        ${SRC_INDEX_DIR}

    # Store key vectors of the sentence datastore.
    python knn_seq/cli/store_keys.py \
        --src-key ${SRC_KEY} \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --num-workers ${NUM_WORKERS} \
        --fp16 \
        --max-tokens 6000 \
        --store-src-sent \
        ${SRC_INDEX_DIR}

Then, build the index of the sentence datastore.

.. code:: bash

    python knn_seq/cli/build_index.py \
        --key-storage ${SRC_INDEX_DIR}/keys.${SRC_KEY}.bin \
        --index-path-prefix ${SRC_INDEX_PATH_PREFIX} \
        --train-size 5242880 \
        --chunk-size 10000000 \
        --metric l2 \
        --hnsw-edges 32 \  # Coarse quantizer to search nearest top-`nprobe` centroids
        --ivf-lists 32768 \  # K-means clustering
        --pq-subvec 64 \  # Product quantization (PQ) to compress the all vectors to uint8 codes.
        --use-opq \  # Rotaion vectors to minimize the PQ error.
        --transform-dim 256  # Reduce the dimension size.

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
        --max-tokens 6000 \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --knn-index-path ${PQ_PATH_PREFIX}.bin \
        --knn-value-path ${INDEX_DIR}/values.bin \
        --knn-key ffn_in \
        --knn-metric l2 \
        --knn-topk 64 \  # The number of nearest neighbors.
        --knn-temperature 100.0 \  # Temperature of kNN softmax.
        --knn-weight 0.5 \  # kNN-MT interpolation parameter.
        --src-key ${SRC_KEY} \
        --src-metric l2 \
        --src-knn-model sentence-transformers/LaBSE \
        --src-topk 512 \  # Search for the 512 nearest neighbor sentences of the input.
        --src-nprobe 64 \
        --src-efsearch 64 \
        --src-index-path ${SRC_INDEX_PATH_PREFIX}.bin \
        --src-value-path ${SRC_INDEX_DIR}/values.bin \
        ${DATABIN_DIR}

   # Case2: NMT encoder
   # Generate.
   fairseq-generate \
        --user-dir knn_seq/ \
        --task translation_knn \
        --fp16 \
        --max-tokens 6000 \
        --path wmt19.de-en.ffn8192/wmt19.de-en.ffn8192.pt \
        --knn-index-path ${PQ_PATH_PREFIX}.bin \
        --knn-value-path ${INDEX_DIR}/values.bin \
        --knn-key ffn_in \
        --knn-metric l2 \
        --knn-topk 64 \  # The number of nearest neighbors.
        --knn-temperature 100.0 \  # Temperature of kNN softmax.
        --knn-weight 0.5 \  # kNN-MT interpolation parameter.
        --src-key ${SRC_KEY} \
        --src-metric l2 \
        --src-topk 512 \  # Search for the 512 nearest neighbor sentences of the input.
        --src-nprobe 64 \
        --src-efsearch 64 \
        --src-index-path ${SRC_INDEX_PATH_PREFIX}.bin \
        --src-value-path ${SRC_INDEX_DIR}/values.bin \
        ${DATABIN_DIR}
