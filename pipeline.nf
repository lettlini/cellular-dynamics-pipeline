workflow {
    input_datasets = Channel.fromPath("${params.in_pdir}/*", type: "dir")
    input_datasets.view { "Input file: ${it}" }
    preprocess(input_datasets)
}
process preprocess {
    publishDir "${params.out_pdir}/${dataset_path.baseName}", mode: 'move'

    input:
    path dataset_path

    output:
    path "preprocessed.pickle", emit: processed_file

    script:
    """
    python ${projectDir}/steps/preprocessing.py \
        --infile="${dataset_path}" \
        --outfile="preprocessed.pickle" \
        --drop_first_n=${params.drop_first_n} \
        --drop_last_m=${params.drop_last_m}
    """
}
