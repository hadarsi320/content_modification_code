python create_bot_features.py --mode=features --index_path=<your-index-path>
--raw_ds_out=<output-for-raw-sentences-dataset> --workingset_file=<output-for-sentence-pairs-workingset> --ref_index=-1 --top_docs_index=3
--doc_tfidf_dir=<path-to-documents-tfidf-vectors-directory>
--sentences_tfidf_dir=<path-to-sentence-tfidf-directory> --queries_file=data/queries_seo_exp.xml --scores_dir=<path-to-ranklib-model-scores-directory>
--trec_file=trecs/trec_file_original_sorted.txt --output_feature_files_dir=<path-to-feature-files-directory>
--output_final_feature_file_dir=<final-features-for-bot-model-directory> --trectext_file=data/documents.trectext
--new_trectext_file=<path-to-modified-documents-trectext-file> --embedding_model_file=<path-to-your-word-embbedings-model-file>--indri_path=<path-to-indri-directory>
--model=rank_models/model_lambdatamart --scripts_path=scripts/ --java_path=<path-to-your-java_path> --jar_path=scripts/RankLib.jar --home_path=~/
--queries_text_file=data/working_comp_queries.txt
--stopwords_file=data/stopwords_list --swig_path=<path-to-your-indri-jar-swig-directory>
