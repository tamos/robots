

# run this to generate all filter implementations with a given parameterization of the pf class


# first for those which we know the initial position

mkdir 'initial_position_known'


for i in $(ls data/pf | grep 'pickle' | grep -v 'simple_world_path2' | grep -v 'rooms_world_path2')
do
	for j in 1 40 80 150 250 300 500 700
	do
		echo $i $j
		python code/pf/RunPF.py 'data/pf/'$i  $j
	done
	# ref https://stackoverflow.com/questions/25577210/grep-a-particular-content-before-a-period
	newdir=$(echo $i | awk -F\. '{print $1}')
	mkdir 'initial_position_known/'$newdir
	mv *.png 'initial_position_known/'$newdir
done 


# now for those we don't

mkdir 'initial_position_unknown'

for i in $(ls data/pf | grep 'pickle' | grep 'path2')
do
	for j in 1 40 80 150 250 300 500 700
	do
		echo $i $j
		python code/pf/RunPF.py 'data/pf/'$i  $j
	done
	# ref https://stackoverflow.com/questions/25577210/grep-a-particular-content-before-a-period
	newdir=$(echo $i | awk -F\. '{print $1}')
	mkdir 'initial_position_unknown/'$newdir
	mv *.png 'initial_position_unknown/'$newdir
done