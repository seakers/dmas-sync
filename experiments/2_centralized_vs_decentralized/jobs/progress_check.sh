#!/bin/bash
# check_progress.sh
# Usage: ./check_progress.sh [results_dir]
# Checks completion status of simulation runs by looking for summary.csv files

RESULTS_DIR="${1:-./results}"
TOTAL_TRIALS=1296

# Check directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: directory '$RESULTS_DIR' not found."
    exit 1
fi

# Count completed trials (those with a summary.csv)
COMPLETED=$(find "$RESULTS_DIR" -mindepth 2 -maxdepth 2 -name "summary.csv" | wc -l)

# Count total output directories (running or completed)
TOTAL_DIRS=$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

# Running = directories without summary.csv
RUNNING=$((TOTAL_DIRS - COMPLETED))

# Percentage complete
if [ "$TOTAL_TRIALS" -gt 0 ]; then
    PCT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED / $TOTAL_TRIALS) * 100}")
else
    PCT=0
fi

# Progress bar (50 chars wide)
FILLED=$(awk "BEGIN {printf \"%d\", ($COMPLETED / $TOTAL_TRIALS) * 50}")
BAR=$(printf '%0.s█' $(seq 1 $FILLED))
EMPTY=$(printf '%0.s░' $(seq 1 $((50 - FILLED))))

echo "============================================================"
echo "  Simulation Progress"
echo "============================================================"
echo "  [${BAR}${EMPTY}] ${PCT}%"
echo ""
echo "  Completed : $COMPLETED / $TOTAL_TRIALS"
echo "  Running   : $RUNNING"
echo "  Remaining : $((TOTAL_TRIALS - COMPLETED))"
echo "============================================================"

# List completed trial names if -v flag passed
if [ "${2}" == "-v" ] || [ "${1}" == "-v" ]; then
    echo ""
    echo "  Completed trials:"
    find "$RESULTS_DIR" -mindepth 2 -maxdepth 2 -name "summary.csv" \
        | sed 's|/summary.csv||' \
        | sed "s|$RESULTS_DIR/||" \
        | sort \
        | awk '{print "    " NR ". " $0}'
fi

