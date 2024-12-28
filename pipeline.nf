params.min_nucleus_area_pxsq = params.min_nucleus_area_mumsq / (params.mum_per_px ** 2)
params.cell_cutoff_px = params.cell_cutoff_mum / params.mum_per_px

workflow {
    input_datasets = Channel.fromPath("${params.in_pdir}", type: "dir")

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


    label_cell_approximation(cell_approximation.out.results)
    label_nuclei_segmentation(nuclei_segmentation.out.results)

    structure_abstraction(label_nuclei_segmentation.out.results, label_cell_approximation.out.results)

    track_cells(label_cell_approximation.out.results, structure_abstraction.out.results)

    build_graphs(track_cells.out.results)
    annotate_graph_theoretical_observables(build_graphs.out.results)
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
    tuple path("cell_approximation.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/cell_approximation.py \
        --infile="${fpath}" \
        --outfile="cell_approximation.pickle" \
        --cell_cutoff_px=${params.cell_cutoff_px}
    """
}

process structure_abstraction {
    publishDir "${params.out_pdir}/${cell_basename}", mode: 'copy'

    input:
    tuple path(nuclei_fpath), val(nuclei_basename)
    tuple path(cell_fpath), val(cell_basename)

    output:
    tuple path("abstract_structure.pickle"), val(nuclei_basename), emit: results

    script:
    """
    python ${projectDir}/steps/structure_abstraction.py \
        --nuclei_infile="${nuclei_fpath}" \
        --cells_infile="${cell_fpath}" \
        --mum_per_px=${params.mum_per_px} \
        --outfile="abstract_structure.pickle"
    """
}

process label_cell_approximation {
    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("cells_labelled.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/label_objects.py \
        --infile=${fpath} \
        --outfile="cells_labelled.pickle"
    """
}

process label_nuclei_segmentation {
    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("nuclei_labelled.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/label_objects.py \
        --infile=${fpath} \
        --outfile="nuclei_labelled.pickle"
    """
}

process track_cells {
    publishDir "${params.out_pdir}/${cell_basename}", mode: 'copy'

    input:
    tuple path(cell_approximation_fpath), val(cell_basename)
    tuple path(abstract_structure_fpath), val(as_basename)

    output:
    tuple path("tracked_abstract_structure.pickle"), val(cell_basename), emit: results

    script:
    """
    python ${projectDir}/steps/track_cells.py \
        --cell_label_file=${cell_approximation_fpath} \
        --abstract_structure_file=${abstract_structure_fpath} \
        --outfile="tracked_abstract_structure.pickle"
    """
}

process build_graphs {
    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    input:
    tuple path(abstract_structure_fpath), val(basename)

    output:
    tuple path("graph_dataset.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/build_graphs.py \
        --infile=${abstract_structure_fpath} \
        --mum_per_px=${params.mum_per_px} \
        --outfile="graph_dataset.pickle"
    """
}

process annotate_graph_theoretical_observables {
    publishDir "${params.out_pdir}/${basename}", mode: 'copy'

    label "high_cpu"

    input:
    tuple path(graph_dataset_fpath), val(basename)

    output:
    tuple path("graph_dataset_annotated.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/graph_theory_annotations.py \
        --infile=${graph_dataset_fpath} \
        --outfile="graph_dataset_annotated.pickle" \
        --cpus=${task.cpus}
    """
}
