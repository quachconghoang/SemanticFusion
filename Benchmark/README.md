Evaluation with TartanAIR dataset
==============================
- _script_data_macro.py_: Download and prepare the TartanAIR dataset (obsoleted because  the authors changed to from Azure to AWS, and the dataset is not available now. We will update the link once it is available again).
- _script_gendata_rosbag.py_: Generate rosbag files with simulated IMUs, cameras from the TartanAIR dataset (workable with OpenVINS - but need some detail instructions for the MSCKF initialize successfully) .

- **Matching benchmarks**
  - _bench_simple_match.py_: A simple benchmark by finding Hungarian algorithm for matching two sets of 2D points.
  - _bench_superglue.py_: Benchmark for SuperGlue matching algorithm
  - _bench_sinkhorn1_match.py_ and bench_sinkhorn2_match.py: Benchmark for your sinkhorn matching-based algorithm