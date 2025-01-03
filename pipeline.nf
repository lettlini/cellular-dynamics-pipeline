params.min_nucleus_area_pxsq = params.min_nucleus_area_mumsq / (params.mum_per_px ** 2)
params.cell_cutoff_px = params.cell_cutoff_mum / params.mum_per_px
params.parent_dir_in = file(params.parent_indir).resolve(params.in_dir).toString()
params.parent_dir_out = file(params.parent_outdir).resolve(params.out_dir).toString()

workflow {
    input_datasets = Channel.fromPath("${params.parent_dir_in}", type: "dir")

    // Transform the channel to emit both the directory and its basename
    // This creates a tuple channel: [dir, basename]
    input_datasets = input_datasets.map { dir ->
        def basename = dir.name
        [dir, basename]
    }

    // input_datasets.view { "Input file: ${it}" }
    prepare_dataset_from_raw(input_datasets)
    confluency_filtering(prepare_dataset_from_raw.out.results)
    nuclei_segmentation(confluency_filtering.out.results)
    cell_approximation(nuclei_segmentation.out.results)


    label_cell_approximation(cell_approximation.out.results)
    label_nuclei_segmentation(nuclei_segmentation.out.results)

    structure_abstraction(label_nuclei_segmentation.out.results, label_cell_approximation.out.results)

    track_cells(label_cell_approximation.out.results, structure_abstraction.out.results)

    build_graphs(track_cells.out.results)
    annotate_graph_theoretical_observables(build_graphs.out.results)
    annotate_neighbor_retention(annotate_graph_theoretical_observables.out.results)
    annotate_D2min(annotate_neighbor_retention.out.results)
}

process prepare_dataset_from_raw {

    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(dataset_path), val(basename)

    output:
    tuple path("original_dataset.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/prepare_dataset.py \
        --indir="${dataset_path}" \
        --outfile="original_dataset.pickle" \
        --provider=${params.provider} \
        --cpus=${task.cpus}
    """
}

process confluency_filtering {

    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(dataset_path), val(basename)

    output:
    tuple path("confluency_filtered.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/filter.py \
        --infile="${dataset_path}" \
        --outfile="confluency_filtered.pickle" \
        --drop_first_n=${params.drop_first_n} \
        --drop_last_m=${params.drop_last_m} \
        --cpus=${task.cpus}
    """
}

process nuclei_segmentation {
    maxForks 1
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

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
        --min_nucleus_area_pxsq=${params.min_nucleus_area_pxsq} \
        --cpus=${task.cpus}
    """
}

process cell_approximation {
    maxForks 2

    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("cell_approximation.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/cell_approximation.py \
        --infile="${fpath}" \
        --outfile="cell_approximation.pickle" \
        --cell_cutoff_px=${params.cell_cutoff_px} \
        --cpus=${task.cpus}
    """
}

process structure_abstraction {
    publishDir "${params.parent_dir_out}/${cell_basename}", mode: 'copy'

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
        --outfile="abstract_structure.pickle" \
        --cpus=${task.cpus}
    """
}

process label_cell_approximation {
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("cells_labelled.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/label_objects.py \
        --infile=${fpath} \
        --outfile="cells_labelled.pickle" \
        --cpus=${task.cpus}
    """
}

process label_nuclei_segmentation {
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(fpath), val(basename)

    output:
    tuple path("nuclei_labelled.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/label_objects.py \
        --infile=${fpath} \
        --outfile="nuclei_labelled.pickle" \
        --cpus=${task.cpus}
    """
}

process track_cells {
    publishDir "${params.parent_dir_out}/${cell_basename}", mode: 'copy'

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
        --outfile="tracked_abstract_structure.pickle" \
        --cpus=${task.cpus}
    """
}

process build_graphs {
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(abstract_structure_fpath), val(basename)

    output:
    tuple path("graph_dataset.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/build_graphs.py \
        --infile=${abstract_structure_fpath} \
        --mum_per_px=${params.mum_per_px} \
        --outfile="graph_dataset.pickle" \
        --cpus=${task.cpus}
    """
}

process annotate_graph_theoretical_observables {
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

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
        --cpus=${task.cpus} \
    """
}
process annotate_neighbor_retention {
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(graph_dataset_fpath), val(basename)

    output:
    tuple path("neighbor_retention_graph_ds.pickle"), val(basename), emit: results

    script:
    """
    python ${projectDir}/steps/annotate_neighbor_retention.py \
        --infile=${graph_dataset_fpath} \
        --outfile="neighbor_retention_graph_ds.pickle" \
        --delta_t_minutes=${params.delta_t_minutes} \
        --lag_times_minutes=${params.lag_times_minutes} \
        --cpus=${task.cpus}
    """
}

process annotate_D2min {
    publishDir "${params.parent_dir_out}/${basename}", mode: 'copy'

    input:
    tuple path(graph_dataset_fpath), val(basename)

    output:
    tuple path("D2min_annotated_graphs.pickle"), val(basename)

    script:
    """
    python ${projectDir}/steps/annotate_D2min.py \
        --infile=${graph_dataset_fpath} \
        --outfile="D2min_annotated_graphs.pickle" \
        --delta_t_minutes=${params.delta_t_minutes} \
        --lag_times_minutes=${params.lag_times_minutes} \
        --mum_per_px=${params.mum_per_px} \
        --minimum_neighbors=${params.minimum_neighbors} \
        --cpus=${task.cpus} \
    """
}
