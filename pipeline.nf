workflow {
    input_datasets = Channel.fromPath("${params.in_pdir}/*", type: "dir")

    // Transform the channel to emit both the directory and its basename
    // This creates a tuple channel: [dir, basename]
    input_datasets = input_datasets.map { dir ->
        def basename = dir.name
        [dir, basename]
    }

    input_datasets.view { "Input file: ${it}" }

    load_and_filter(input_datasets)
    preprocess(
        load_and_filter.out.results
    )
}

process load_and_filter {

    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(dataset_path), val(basename)

    output:
    tuple path("original_filtered.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/filter.py \
        --infile="${dataset_path}" \
        --outfile="original_filtered.pickle" \
        --drop_first_n=${params.drop_first_n} \
        --drop_last_m=${params.drop_last_m}
    """
}

process preprocess {

    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("preprocessed.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/preprocessing.py \
        --infile="${fpath}" \
        --outfile="preprocessed.pickle" \
    """
}
