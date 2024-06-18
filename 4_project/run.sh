for t in 1 2 4 6 8
do
    for s in 1000 2000 4000 8000 10000 20000 40000 80000 100000 200000 400000 800000 1000000 2000000 4000000 8000000
    do
        g++ -DNUMT=$t -DARRAYSIZE=$s project04.cpp -o proj04 -lm -fopenmp
        ./proj04
    done
    echo ""  
done
