include { data_preparation } from './data-preparation/pipeline.nf'
//include { data_analysis    } from './data-analysis/pipeline.nf'

workflow {

    input_datasets = Channel.fromPath(file(params.parent_indir).resolve(params.in_dir).toString(), type: "dir")
    // Transform the channel to emit both the directory and its basename
    // This creates a tuple channel: [dir, basename]
    input_datasets = input_datasets.map { dir ->
        def basename = dir.name
        [basename, dir]
    }
        | data_preparation
}
