include { data_preparation } from './data-preparation/pipeline.nf'
include { data_analysis    } from './data-analysis/pipeline.nf'

workflow {
    config_files = Channel.fromPath(file(params.parent_indir).resolve(params.in_dir).toString() + "/**/config.toml", type: "file")

    parent_directories = config_files.flatMap { cfile ->
        def dir = cfile.parent
        def subdirs = dir.listFiles().findAll { it.isDirectory() }
        // assuming the parent directory name is the config type
        subdirs.collect { subdir ->
            [subdir.name, subdir, cfile]
        }
    }
    if (params.test == true) {
        // Only executes if params.test is explicitly true
        println("Test mode is enabled")
        parent_directories = parent_directories.take(3)
    }
    else {
        println("Test mode is disabled")
    }

    parent_directories
        | data_preparation
        | data_analysis
}
