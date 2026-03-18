OUTPUT_DIR=./nextflow-reports
mkdir -p $OUTPUT_DIR  # Create the output directory if it doesn't exist

CONFIG_ID=$1

nextflow run ./pipeline-umbrella.nf \
    -c ./dataset_configs/$CONFIG_ID.config \
    -profile local \
    -with-report ${OUTPUT_DIR}/report.html \
    -with-timeline ${OUTPUT_DIR}/timeline.html \
    -with-trace ${OUTPUT_DIR}/trace.txt \
    "${@:2}"
