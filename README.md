# content_modification_code
Code for the paper "Ranking-Incentivized Quality Preserving Content Modification"


# Requirements
To run the code, follwoing packages are required

1) Indri 5.6

2) gensim

3) nltk

4) javaobj

5) Python 3.5 (or newer), Java 1.8

The code is intended to create qrels/features file for bot model creation.

There are 2 modes for running the code. The first one is "qrels" mode. In this mode for every pair of sentences in the dataset, a modification in the promoted document is created and the list is reranked accordigly. Then it is possible to obtain the rank-promotion label - in this mode only rank-promotion labels are created. The second mode is "features". In this mode features file is created containing all sentence pairs according to the dataset.


# Example run command
An example for a running command is at "create_features_command.sh" file. In order to run it, modify the input paramters according to your own file system.


# Parameters
--mode={features/qrels}

--index_path=\<your-indri-index-path\>
  
--raw_ds_out=\<output-for-raw-sentences-dataset\>
  
--workingset_file=\<output-for-sentence-pairs-workingset\>
  
--ref_index=-1 -\> according to initial retrieved list, the index of documents in the list chosen for rank promotion

--top_docs_index=3 -\> number of top-retrieved documents for candidate set creation 

--doc_tfidf_dir=\<path-to-documents-tfidf-vectors-directory\>
  
--sentences_tfidf_dir=\<path-to-sentence-tfidf-directory\> -\> the code creates tf-idf represantation of sentences 
  
--queries_file=data/queries_seo_exp.xml -\> xml of queries
  
--scores_dir=\<path-to-ranklib-model-scores-directory\> -\> scores directory for reranking
  
--trec_file=trecs/trec_file_original_sorted.txt -\> original trec file (original ranked lists induced from this file)

--output_feature_files_dir=\<path-to-feature-files-directory\>
  
--output_final_feature_file_dir=\<final-features-for-bot-model-directory\>
  
--trectext_file=data/documents.trectext -\> xml file containing trectext format of original documents

--new_trectext_file=\<path-to-modified-documents-trectext-file\> -\> path to the modified trectext file (after replacing sentences)
  
--embedding_model_file=\<path-to-your-word-embbedings-model-file\>
  
--indri_path=\<path-to-indri-directory\>
  
--model=rank_models/model_lambdatamart

--scripts_path=scripts/

--java_path=\<path-to-your-java_path\> 

--jar_path=scripts/RankLib.jar 

--home_path=~/

--queries_text_file=data/working_comp_queries.txt

--stopwords_file=data/stopwords_list 

--swig_path=\<path-to-your-indri-jar-swig-directory\>
