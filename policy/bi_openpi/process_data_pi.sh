# bash process_data_pi.sh block_hammer_beat D435 100

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}

cd ../..
python script/pkl2hdf5_pi.py $task_name $head_camera_type $expert_data_num