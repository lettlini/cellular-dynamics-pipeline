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
include { calculate_local_density                } from './cellular-dynamics-nf-modules/modules/graph_processing/calculate_local_density/main.nf'
include { concatenate_tracking_dataframes        } from './cellular-dynamics-nf-modules/modules/tracking/concatenate_tracking_dataframes/main.nf'
include { nucleus_displacement_index             } from './cellular-dynamics-nf-modules/modules/image_processing/nucleus_displacement_index/main.nf'
include { cage_relative_squared_displacement     } from './cellular-dynamics-nf-modules/modules/tracking/cage_relative_squared_displacement/main.nf'

workflow data_preparation {
    take:
    input_datasets

    main:

    parent_dir_out = Channel.value(file(params.parent_outdir_preparation).resolve(params.out_dir).toString())
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
    nucleus_displacement_index(
        label_nuclei.out.results.join(label_cells.out.results, failOnDuplicate: true, failOnMismatch: true).join(build_graphs.out.results, failOnDuplicate: true, failOnMismatch: true),
        parent_dir_out,
    )

    calculate_local_density(nucleus_displacement_index.out.results, parent_dir_out).collect()
    annotate_graph_theoretical_observables(calculate_local_density.out.results, parent_dir_out)
    annotate_neighbor_retention(annotate_graph_theoretical_observables.out.results, params.delta_t_minutes, params.lag_times_minutes, parent_dir_out)
    cage_relative_squared_displacement(annotate_neighbor_retention.out.results, params.delta_t_minutes, params.lag_times_minutes, params.mum_per_px, parent_dir_out)
    all_graph_datasets = annotate_D2min(cage_relative_squared_displacement.out.results, params.delta_t_minutes, params.lag_times_minutes, params.mum_per_px, params.minimum_neighbors, parent_dir_out)

    // dataframe
    assemble_cell_track_dataframe(annotate_D2min.out.results, params.delta_t_minutes, params.include_attrs, params.exclude_attrs, parent_dir_out)
    all_dataframes_list = add_cell_culture_metadata(assemble_cell_track_dataframe.out.results, params.provider, parent_dir_out).collect { _first, second -> second }
    concatenate_tracking_dataframes(all_dataframes_list, parent_dir_out)

    emit:
    all_cell_tracks_dataframe = concatenate_tracking_dataframes.out.results
    all_graph_datasets        = all_graph_datasets // this is a list of tuples of the form [basename, path]
}

process prepare_dataset_from_raw {

    publishDir "${parent_dir_out}/${basename}", mode: 'copy'

    label "low_cpu", "short_running"

    conda "${moduleDir}/environment.yml"

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
    python ${moduleDir}/scripts/prepare_dataset.py \
        --indir="${dataset_path}" \
        --outfile="original_dataset.pickle" \
        --provider=${provider} \
        --cpus=${task.cpus}
    """
}

process add_cell_culture_metadata {

    publishDir "${parent_dir_out}/${basename}", mode: 'copy'

    label "low_cpu", "short_running"

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(basename), path(cell_track_df_path)
    val provider
    val parent_dir_out

    output:
    tuple val(basename), path("cell_tracks_with_metadata.ipc"), emit: results

    script:
    """
    python ${moduleDir}/scripts/add_dataset_metadata.py \
        --infile="${cell_track_df_path}" \
        --outfile="cell_tracks_with_metadata.ipc" \
        --basename=${basename} \
        --provider=${provider} \
        --cpus=${task.cpus}
    """
}
