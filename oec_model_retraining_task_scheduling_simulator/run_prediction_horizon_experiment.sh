echo "Running prediction horizon experiments"
echo "Experiment with constellation: starlink_top10_gs"
python run_vary_predict_horizon.py -c starlink_top10_gs -p 10,15,20,25,30,35,40 -r 0,1,2
echo "Experiment with constellation: sentinel2_sentinel_gs"
python run_vary_predict_horizon.py -c sentinel2_sentinel_gs -p 1,3,5,7,9,11,13 -r 0,1,2
echo "Finished running prediction horizon experiments"