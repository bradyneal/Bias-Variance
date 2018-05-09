cmd="python3 ngc_run.py $@"
echo $cmd

ngc batch run \
  --instance ngcv1 \
  --name "Test job" \
  --image "mila1234/tantiavi-pytorch-volta:1" \
  --result /data/milatmp1/root/information-paths/saved/ \
  --command "git clone git@github.com:bradyneal/information-paths.git; \
    cd information-paths; \
    pip install -r requirements.txt; \
    $cmd"
