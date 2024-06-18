module load openmpi
mpic++ proj07.cpp -o proj07 -lm
for n in 1 2 4 6 8
do
    mpiexec -mca btl self,tcp -np $n ./proj07
done
