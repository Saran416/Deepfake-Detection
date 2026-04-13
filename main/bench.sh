CHECKPOINT="./test_celeb/best_model_xception.pth"
VIDEO_DIR="./test_videos"
WARMUP_RUNS=10
TIMED_RUNS=100
OUTPUT_JSON="results_xception.json"

python benchmark.py \
    --checkpoint "$CHECKPOINT" \
    --video_dir "$VIDEO_DIR" \
    --warmup_runs "$WARMUP_RUNS" \
    --timed_runs "$TIMED_RUNS" \
    --output_json "$OUTPUT_JSON"