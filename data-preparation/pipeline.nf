include { label_objects                          } from './cellular-dynamics-nf-modules/modules/image_processing/label_objects/main.nf'
include { nuclei_segmentation                    } from './cellular-dynamics-nf-modules/modules/image_processing/nuclei_segmentation/main.nf'
include { confluency_filter                      } from './cellular-dynamics-nf-modules/modules/image_processing/confluency_filter/main.nf'
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
include { cage_relative_squared_displacement     } from './cellular-dynamics-nf-modules/modules/tracking/cage_relative_squared_displacement/main.nf'

workflow data_preparation {
    take:
    input_datasets

    main:
    publish_dir = params.parent_outdir_preparation + params.out_dir

    // clear the parent_outdir
    new File(publish_dir).deleteDir()
    new File(publish_dir).mkdirs()

    prepare_dataset_from_raw(input_datasets, publish_dir)
    nuclei_segmentation(prepare_dataset_from_raw.out.results, params.min_nucleus_area_mumsq, publish_dir)


    confluency_filter(nuclei_segmentation.out.results, "nuclei", publish_dir)


    label_nuclei(confluency_filter.out.results, "nuclei", publish_dir)
    cell_approximation(confluency_filter.out.results, params.cell_cutoff_mum, publish_dir)
    label_cells(cell_approximation.out.results, "cells", publish_dir)

    structure_abstraction_input = label_nuclei.out.results
        .join(label_cells.out.results, by: [0], failOnDuplicate: true, failOnMismatch: true)
        .map {
            // (0: id, 1: file1, 2: conf1, 3: file2, 4: conf2)
            tuple(
                it[0],
                it[1],
                it[3],
                it[2],
            )
        }

    structure_abstraction(structure_abstraction_input, publish_dir)

    cell_tracking_overlap_input = label_cells.out.results.join(structure_abstraction.out.results, by: [0], failOnDuplicate: true, failOnMismatch: true)
        | map {
            tuple(
                it[0],
                it[1],
                it[3],
                it[2],
            )
        }

    cell_tracking_overlap(cell_tracking_overlap_input, publish_dir)
    build_graphs(cell_tracking_overlap.out.results, publish_dir)

    annotate_D2min(build_graphs.out.results, params.lag_times_minutes, params.minimum_neighbors, publish_dir)
    cage_relative_squared_displacement(annotate_D2min.out.results, params.lag_times_minutes, publish_dir)

    // graph dataset
    k_order_density(cage_relative_squared_displacement.out.results, publish_dir)
    all_graph_datasets = calculate_local_density(k_order_density.out.results, publish_dir).collect()

    // dataframe
    assemble_cell_track_dataframe(calculate_local_density.out.results, params.include_attrs, params.exclude_attrs, publish_dir)
    all_dataframes_list = add_cell_culture_metadata(assemble_cell_track_dataframe.out.results, publish_dir).collect { _first, second, _third -> second }
    concatenate_tracking_dataframes(all_dataframes_list, publish_dir)

    emit:
    all_cell_tracks_dataframe = concatenate_tracking_dataframes.out.results
    all_graph_datasets        = all_graph_datasets // this is a list of tuples of the form [basename, file, config]
}

process prepare_dataset_from_raw {

    publishDir "${publish_directory}/${basename}", mode: 'copy'

    label "low_cpu", "short_running"

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(basename), path(dataset_path), path(dataset_config)
    val publish_directory

    output:
    tuple val(basename), path("original_dataset.pickle"), path(dataset_config), emit: results

    script:
    """
    echo "Processing: ${basename}"
    echo "Dataset Path: ${dataset_path}, Basename: ${basename}"
    python ${moduleDir}/scripts/prepare_dataset.py \
        --indir="${dataset_path}" \
        --outfile="original_dataset.pickle" \
        --dataset_config="${dataset_config}" \
        --cpus=${task.cpus}
    """
}

process add_cell_culture_metadata {

    publishDir "${parent_dir_out}/${basename}", mode: 'copy'

    label "low_cpu", "short_running"

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(basename), path(cell_track_df_path), path(dataset_config)
    val parent_dir_out

    output:
    tuple val(basename), path("cell_tracks_with_metadata.ipc"), path(dataset_config), emit: results

    script:
    """
    python ${moduleDir}/scripts/add_dataset_metadata.py \
        --infile="${cell_track_df_path}" \
        --outfile="cell_tracks_with_metadata.ipc" \
        --basename=${basename} \
        --dataset_config=${dataset_config} \
        --cpus=${task.cpus}
    """
}

process k_order_density {

    publishDir "${parent_dir_out}/${basename}", mode: 'copy'

    label "high_cpu", "long_running"

    conda "${moduleDir}/environment.yml"

    input:
    tuple val(basename), path(graph_ds), path(dataset_config)
    val parent_dir_out

    output:
    tuple val(basename), path("graph_dataset_with_k_order_density.pickle"), path(dataset_config), emit: results

    script:
    """
    python ${moduleDir}/scripts/k_order_density.py \
        --infile="${graph_ds}" \
        --outfile="graph_dataset_with_k_order_density.pickle" \
        --basename=${basename} \
        --dataset_config=${dataset_config} \
        --cpus=${task.cpus}
    """
}
