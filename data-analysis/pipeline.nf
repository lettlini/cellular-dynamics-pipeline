workflow data_analysis {
    take:
    all_cell_tracks_dataframe
    all_graph_datasets

    main:
    parent_dir_out = file(params.parent_outdir_analysis).resolve(params.out_dir).toString()

    // clear the parent_outdir
    new File(parent_dir_out).deleteDir()
    new File(parent_dir_out).mkdirs()

    python_files_ch = Channel.fromPath("${moduleDir}/analysis-scripts/**/*.py", hidden: false)

    execute_python_analysis_script(python_files_ch, all_cell_tracks_dataframe, all_graph_datasets, parent_dir_out)
}

process execute_python_analysis_script {

    publishDir "${parent_dir_out}/${python_file.baseName}", mode: 'copy'

    label "single_threaded", "short_running"

    conda "${moduleDir}/environment.yml"

    input:
    path python_file
    path all_cell_tracks_dataframe
    val all_graph_datasets
    val parent_dir_out

    output:
    path "*"

    script:
    """
    python ${python_file} \
        --dataframe_file=${all_cell_tracks_dataframe} \
        --parent_dir_out="." \
    """
}
