# 256 particles
# in a box (size L) so density = 0.4 = N*pi*a*b / L^2
# Major axis length 2
# Minor axis length 1
# in aa periodic box
# simulate 30 seconds of data
# at framerate 1.0/600.0
# save every 10th frame
# "collision strength" 300
# particle self propulsion speed 5
 ./CUDAABP -N 256 --initial-packing-fraction 0.4 -a 2 -b 1 -periodic 1 -T 30 -dt 0.001666 -mu 300 -v 10 --save-every 10

julia render.jl --radius 2 --minor-axis 1
