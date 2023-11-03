python serverrun_simulation.py $1 &
sleep 20
python clientrun_simulation.py 0 $1 &
python clientrun_simulation.py 1 $1 > /dev/null 2>&1 &
python clientrun_simulation.py 2 $1 > /dev/null 2>&1 &
python clientrun_simulation.py 3 $1 > /dev/null 2>&1 &
python clientrun_simulation.py 4 $1 > /dev/null 2>&1 &