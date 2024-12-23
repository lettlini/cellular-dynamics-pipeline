params.min_nucleus_area_pxsq = params.min_nucleus_area_mumsq / (params.mum_per_px ** 2)
params.cell_cutoff_px = params.cell_cutoff_mum / params.mum_per_px

workflow {
    input_datasets = Channel.fromPath("${params.in_pdir}/*", type: "dir")

    // Transform the channel to emit both the directory and its basename
    // This creates a tuple channel: [dir, basename]
    input_datasets = input_datasets.map { dir ->
        def basename = dir.name
        [dir, basename]
    }

    // input_datasets.view { "Input file: ${it}" }

    load_and_filter(input_datasets)
    preprocess(
        load_and_filter.out.results
    )
    nuclei_segmentation(preprocess.out.results).collect(flat: false)
    cell_approximation(nuclei_segmentation.out.results)
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

process nuclei_segmentation {
    maxForks 1
    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("nuclei_segmentation.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/nuclei_segmentation.py \
        --infile="${fpath}" \
        --outfile="nuclei_segmentation.pickle" \
        --stardist_probility_threshold=${params.stardist_probality_threshold} \
        --min_nucleus_area_pxsq=${params.min_nucleus_area_pxsq}
    """
}

process cell_approximation {
    maxForks 2

    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("cell_approximation.pickle"), val(basename), emit: result

    script:
    """
    python ${projectDir}/steps/cell_approximation.py \
        --infile="${fpath}" \
        --outfile="cell_approximation.pickle" \
        --cell_cutoff_px=${params.cell_cutoff_px}
    """
}
