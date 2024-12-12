#! /usr/bin/bash

cd Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "1 iteration done"
cd ../Baseline4/
python3 run.py
echo "1 iteration done"
cd ../Baseline1/
python3 run.py
echo "1 iteration done"
cd ../Baseline2/
python3 run.py
echo "1 iteration done"
cd ../Baseline3/
python3 run.py
echo "1 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "2 iteration done"
cd ../Baseline4/
python3 run.py
echo "2 iteration done"
cd ../Baseline1/
python3 run.py
echo "2 iteration done"
cd ../Baseline2/
python3 run.py
echo "2 iteration done"
cd ../Baseline3/
python3 run.py
echo "2 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "3 iteration done"
cd ../Baseline4/
python3 run.py
echo "3 iteration done"
cd ../Baseline1/
python3 run.py
echo "3 iteration done"
cd ../Baseline2/
python3 run.py
echo "3 iteration done"
cd ../Baseline3/
python3 run.py
echo "3 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "4 iteration done"
cd ../Baseline4/
python3 run.py
echo "4 iteration done"
cd ../Baseline1/
python3 run.py
echo "4 iteration done"
cd ../Baseline2/
python3 run.py
echo "4 iteration done"
cd ../Baseline3/
python3 run.py
echo "4 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "5 iteration done"
cd ../Baseline4/
python3 run.py
echo "5 iteration done"
cd ../Baseline1/
python3 run.py
echo "5 iteration done"
cd ../Baseline2/
python3 run.py
echo "5 iteration done"
cd ../Baseline3/
python3 run.py
echo "5 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "6 iteration done"
cd ../Baseline4/
python3 run.py
echo "6 iteration done"
cd ../Baseline1/
python3 run.py
echo "6 iteration done"
cd ../Baseline2/
python3 run.py
echo "6 iteration done"
cd ../Baseline3/
python3 run.py
echo "6 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "7 iteration done"
cd ../Baseline4/
python3 run.py
echo "7 iteration done"
cd ../Baseline1/
python3 run.py
echo "7 iteration done"
cd ../Baseline2/
python3 run.py
echo "7 iteration done"
cd ../Baseline3/
python3 run.py
echo "7 iteration done"

cd ../Common/
python3 generate_mobility_matrix.py
cd ../MainAlgo/
python3 run.py
echo "8 iteration done"
cd ../Baseline4/
python3 run.py
echo "8 iteration done"
cd ../Baseline1/
python3 run.py
echo "8 iteration done"
cd ../Baseline2/
python3 run.py
echo "8 iteration done"
cd ../Baseline3/
python3 run.py
echo "8 iteration done"

# cd ../Common/
# python3 generate_mobility_matrix.py
# cd ../MainAlgo/
# python3 run.py
# echo "9 iteration done"
# cd ../Baseline4/
# python3 run.py
# echo "9 iteration done"
# cd ../Baseline1/
# python3 run.py
# echo "9 iteration done"
# cd ../Baseline2/
# python3 run.py
# echo "9 iteration done"
# cd ../Baseline3/
# python3 run.py
# echo "9 iteration done"

# cd ../Common/
# python3 generate_mobility_matrix.py
# cd ../MainAlgo/
# python3 run.py
# echo "10 iteration done"
# cd ../Baseline4/
# python3 run.py
# echo "10 iteration done"
# cd ../Baseline1/
# python3 run.py
# echo "10 iteration done"
# cd ../Baseline2/
# python3 run.py
# echo "10 iteration done"
# cd ../Baseline3/
# python3 run.py
# echo "10 iteration done"

cd ..
python3 coalate_results.py