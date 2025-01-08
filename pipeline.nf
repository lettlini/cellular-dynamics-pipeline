include { label_objects                          } from './cellular-dynamics-nf-modules/modules/image_processing/label_objects/main.nf'
include { nuclei_segmentation                    } from './cellular-dynamics-nf-modules/modules/image_processing/nuclei_segmentation/main.nf'
include { basic_filter as confluency_filter      } from './cellular-dynamics-nf-modules/modules/image_processing/basic_filter/main.nf'
include { cell_approximation                     } from './cellular-dynamics-nf-modules/modules/image_processing/cell_approximation/main.nf'
include { label_objects as label_nuclei ; label_objects as label_cells } from './cellular-dynamics-nf-modules/modules/image_processing/label_objects/main.nf'
include { structure_abstraction                  } from './cellular-dynamics-nf-modules/modules/graph_processing/structure_abstraction/main.nf'
include { cell_tracking_overlap                  } from './cellular-dynamics-nf-modules/modules/tracking/cell_tracking_overlap/main.nf'
include { build_graphs                           } from './cellular-dynamics-nf-modules/modules/graph_processing/build_graphs/main.nf'
include { annotate_graph_theoretical_observables } from './cellular-dynamics-nf-modules/modules/graph_processing/annotate_graph_theoretical_observables/main.nf'
include { annotate_neighbor_retention            } from './cellular-dynamics-nf-modules/modules/tracking/annotate_neighbor_retention/main.nf'
include { annotate_D2min                         } from './cellular-dynamics-nf-modules/modules/tracking/annotate_D2min/main.nf'
include { assemble_cell_track_dataframe          } from './cellular-dynamics-nf-modules/modules/tracking/assemble_cell_tracks_dataframe/main.nf'

workflow {

    input_datasets = Channel.fromPath(file(params.parent_indir).resolve(params.in_dir).toString(), type: "dir")

    // Transform the channel to emit both the directory and its basename
    // This creates a tuple channel: [dir, basename]
    input_datasets = input_datasets.map { dir ->
        def basename = dir.name
        [basename, dir]
    }

    parent_dir_out = Channel.value(file(params.parent_outdir).resolve(params.out_dir).toString())
    min_nucleus_area_pxsq = Channel.value(params.min_nucleus_area_mumsq / (params.mum_per_px ** 2))
    cell_cutoff_px = Channel.value(params.cell_cutoff_mum / params.mum_per_px)

    prepare_dataset_from_raw(input_datasets, params.provider, parent_dir_out)
    confluency_filter(prepare_dataset_from_raw.out.results, params.drop_first_n, params.drop_last_m, parent_dir_out)
    nuclei_segmentation(
        confluency_filter.out.results,
        params.stardist_probality_threshold,
        min_nucleus_area_pxsq,
        parent_dir_out,
    )
    cell_approximation(nuclei_segmentation.out.results, cell_cutoff_px, parent_dir_out)
    label_nuclei(nuclei_segmentation.out.results, "nuclei", parent_dir_out)
    label_cells(cell_approximation.out.results, "cells", parent_dir_out)

    structure_abstraction(
        label_nuclei.out.results.join(label_cells.out.results, by: [0], failOnDuplicate: true, failOnMismatch: true),
        params.mum_per_px,
        parent_dir_out,
    )

    cell_tracking_overlap(label_cells.out.results.join(structure_abstraction.out.results, by: [0], failOnDuplicate: true, failOnMismatch: true), parent_dir_out)
    build_graphs(cell_tracking_overlap.out.results, params.mum_per_px, parent_dir_out)
    annotate_graph_theoretical_observables(build_graphs.out.results, parent_dir_out)
    annotate_neighbor_retention(annotate_graph_theoretical_observables.out.results, params.delta_t_minutes, params.lag_times_minutes, parent_dir_out)
    annotate_D2min(annotate_neighbor_retention.out.results, params.delta_t_minutes, params.lag_times_minutes, params.mum_per_px, params.minimum_neighbors, parent_dir_out)
    assemble_cell_track_dataframe(annotate_D2min.out.results, params.delta_t_minutes, params.include_attrs, params.exclude_attrs, parent_dir_out)
}

process prepare_dataset_from_raw {

    publishDir "${parent_dir_out}/${basename}", mode: 'copy'

    label "low_cpu", "short_running"

    input:
    tuple val(basename), path(dataset_path)
    val provider
    val parent_dir_out

    output:
    tuple val(basename), path("original_dataset.pickle"), emit: results

    script:
    """
    echo "Processing: ${basename}"
    echo "Dataset Path: ${dataset_path}, Basename: ${basename}"
    python ${projectDir}/scripts/prepare_dataset.py \
        --indir="${dataset_path}" \
        --outfile="original_dataset.pickle" \
        --provider=${provider} \
        --cpus=${task.cpus}
    """
}
